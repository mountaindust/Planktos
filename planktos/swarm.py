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
from scipy.spatial import distance
import pandas as pd
if sys.platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.collections import LineCollection
from matplotlib.path import Path

from planktos import environment, dataio, motion, geom

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class swarm:
    '''
    Fundamental Planktos object describing a group of similar agents.

    The swarm class (alongside the environment class) provides the main agent 
    functionality of Planktos. Each swarm object should be thought of as a group 
    of similar (though not necessarily identical) agents. Planktos implements 
    agents in this way rather than as individual objects for speed purposes; it 
    is easier to vectorize a swarm of agents than individual agent objects, and 
    also much easier to plot them all, get data on them all, etc.

    The swarm object contains all information on the agents' positions, 
    properties, and movement algorithm. It also handles plotting of the agents 
    and saving of agent data to file for further analysis.

    Initial agent velocities will be set as the local fluid velocity if present,
    otherwise zero. Assignment to the velocities attribute can be made directly 
    for other initial conditions.

    NOTE: To customize agent behavior, subclass this class and re-implement the
    method get_positions (do not change the call signature).

    Parameters
    ----------
    swarm_size : int, default=100
        Number of agents in the swarm. ignored when using the 'grid' init method. 
    envir : environment object, optional
        Environment for the swarm to exist in. Defaults to a newly initialized 
        environment with all of the defaults.
    init : {'random', 'grid', ndarray}, default='random'
        Method for initalizing agent positions.
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
    **kwargs : dict, optional
        keyword arguments to be used in the 'grid' initialization method or
        values to be set as a swarm object property. In the latter case, these 
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
        swarm.grid_init for more information.

    Attributes
    ----------
    envir : environment object
        environment that this swarm belongs to
    positions : masked array, shape Nx2 (2D) or Nx3 (3D)
        spatial location of all the agents in the swarm. the mask is False for 
        any row corresponding to an agent that is within the spatial boundaries 
        of the environment, otherwise the mask for the row is set to True and 
        the position of that agent is no longer updated
    N : read only property, int
        The current number of agents in the swarm, based on positions.shape[0]
    pos_history : list of masked arrays
        all previous position arrays are stored here. to get their corresponding 
        times, check the time_history attribute of the swarm's environment.
    full_pos_history : list of masked arrays
        same as pos_history, but also includes the positions attribute as the 
        last entry in the list
    velocities : masked array, shape Nx2 (2D) or Nx3 (3D)
        velocity of all the agents in the swarm. same masking properties as 
        positions
    vel_history : list of masked arrays
        all previous velocity arrays are stored here. to get their corresponding 
        times, check the time_history attribute of the swarm's environment.
    full_vel_history : list of masked arrays
        same as vel_history, but also includes the velocities attribute as the 
        last entry in the list
    accelerations : masked array, shape Nx2 (2D) or Nx3 (3D)
        accelerations of all the agents in the swarm. same masking properties as 
        positions
    ib_collision : 1D array of bool with length equal to the swarm size
        For each agent, True if the agent collided with an immersed boundary in
        the most recent time it moved. False otherwise.
    props : pandas DataFrame
        Pandas dataframe of individual agent properties that vary between agents. 
        This is the method by which individual variation among the agents should 
        be specified. A special column can be specified called 'angle' which, if 
        props_history is being stored, will plot as the agent heading in 2D.
    props_history : List of past Pandas DataFrames or None
        If not None, this list records individual agent attributes at all 
        previous points in time corresponding to the time_history attribute of 
        the swarm's environment.
    full_props_history : List of Pandas DataFrames or None
        props_history plus the current time version of props
    shared_props : dictionary
        dictionary of properties shared by all agents as name-value pairs. Must 
        include 'name' and 'color' indicating the name of the swarm and its 
        default color for plotting. 'mu' and 'cov' are required for Brownian 
        motion, and other properties may be required for other physics.
    rndState : numpy Generator object
        random number generator for this swarm, seeded by the "seed" parameter

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

    In order to accomodate general, user-defined behavior algorithms, all other 
    agent behaviors should be explicitly specified by subclassing this swarm 
    class and overriding the get_positions method. This is easy, and takes the 
    following form: ::

        class myagents(planktos.swarm):
            
            def get_positions(self, dt, params=None):
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
                #   access to all of the swarm attributes via the "self" 
                #   argument. For example, self.positions will return an NxD 
                #   masked array of current agent positions. The one thing this 
                #   method SHOULD NOT do is set the positions, velocities, or 
                #   accelerations attributes of the swarm. This will be handled 
                #   automatically after this method returns, and after boundary 
                #   conditions have been checked.

                return newpositions

    Then, when you create a swarm object, create it using::
    
        swrm = myagents() # add swarm parameters as necessary, as documented above

    This will create a swarm object, but with your my_positions method instead 
    of the default one!

    Examples
    --------
    Create a default swarm in an environment with some fluid data loaded and tiled.

    >>> envir = planktos.environment()
    >>> envir.read_IBAMR3d_vtk_dataset('../tests/IBAMR_test_data', start=5, finish=None)
    >>> envir.tile_flow(3,3)
    >>> swrm = swarm(envir=envir)

    '''

    def __init__(self, swarm_size=100, envir=None, init='random', 
                 ib_condition='sliding', seed=None, shared_props=None, 
                 props=None, store_prop_history=False, name='organism', 
                 color='darkgreen', **kwargs):

        # use a new, 3D default environment if one was not given. Or infer
        #   dimension from init if possible.
        if envir is None:
            if isinstance(init,str):
                self.envir = environment(init_swarms=self, Lz=10)
            elif isinstance(init,np.ndarray) and len(init.shape) == 2:
                if init.shape[1] == 2:
                    self.envir = environment(init_swarms=self)
                else:
                    self.envir = environment(init_swarms=self, Lz=10)
            else:
                if len(init) == 2:
                    self.envir = environment(init_swarms=self)
                else:
                    self.envir = environment(init_swarms=self, Lz=10)
        else:
            try:
                assert envir.__class__.__name__ == 'environment'
                envir.swarms.append(self)
                self.envir = envir
            except AssertionError as ae:
                print("Error: invalid environment object.")
                raise ae

        # initialize random number generator
        self.rndState = np.random.default_rng(seed=seed)

        # initialize agent locations
        if isinstance(init,np.ndarray) and len(init.shape) == 2:
            swarm_size = init.shape[0]
        self.positions = ma.zeros((swarm_size, len(self.envir.L)))
        if isinstance(init,str):
            if init == 'random':
                print('Initializing swarm with uniform random positions...')
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
                print('Initializing swarm with grid positions...')
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

        # Initialize IB collision detection
        self.ib_collision = np.full(swarm_size, False)
        self.ib_condition = ib_condition

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

        # Record any kwargs as swarm parameters
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
        automatically be called by the environment class's calculate_FTLE method, 
        and if you want to initialize a swarm with a grid this is possible by 
        passing the init='grid' keyword argument when the swarm is created. 
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
        This algorithm is meant as a huristic only! It is not guaranteed to mask 
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
                intersections = geom.seg_intersect_2D(pt, endpt,
                    close_mesh[:,0,:], close_mesh[:,1,:], get_all=True)
            else:
                intersections = geom.seg_intersect_3D_triangles(pt, endpt,
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
        * Each subsequent row corresponds to a different agent in the swarm.
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
                dataio.write_vtk_point_data(path, name, data)
            else:
                dataio.write_vtk_point_data(path, name, self.positions[~ma.getmaskarray(self.positions[:,0]),:])
        else:
            for cyc, time in enumerate(self.envir.time_history):
                if DIM2:
                    data = np.zeros((self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:].shape[0],3))
                    data[:,:2] = self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:].squeeze()
                    dataio.write_vtk_point_data(path, name, data, 
                                                 cycle=cyc, time=time)
                else:
                    dataio.write_vtk_point_data(path, name, 
                        self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:].squeeze(), 
                        cycle=cyc, time=time)
            cyc = len(self.envir.time_history)
            if DIM2:
                data = np.zeros((self.positions[~ma.getmaskarray(self.positions[:,0]),:].shape[0],3))
                data[:,:2] = self.positions[~ma.getmaskarray(self.positions[:,0]),:]
                dataio.write_vtk_point_data(path, name, data, cycle=cyc,
                                             time=self.envir.time)
            else:
                dataio.write_vtk_point_data(path, name, 
                    self.positions[~ma.getmaskarray(self.positions[:,0]),:],
                    cycle=cyc, time=self.envir.time)



    def _change_envir(self, envir):
        '''Manages a change from one environment to another'''

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
                raise RuntimeError("Swarm dimension smaller than new environment dimension!"+
                    " Cannot scale up!")
        self.envir = envir
        envir.swarms.append(self)



    def calc_re(self, u, diam=None):
        '''Calculate and return the Reynolds number as experienced by a swarm 
        with characteristic length 'diam' in a fluid moving with velocity u. All 
        other parameters will be pulled from the environment's attributes. 
        
        If diam is not specified, this method will look for it in the 
        shared_props dictionary of this swarm.
        
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
            raise RuntimeError("Parameters necessary for Re calculation in environment are undefined.")



    def move(self, dt=1.0, params=None, ib_collisions='default', 
             update_time=True, silent=False):
        '''Move all organisms in the swarm over one time step of length dt.
        DO NOT override this method when subclassing; override get_positions
        instead!!!

        Performs a lot of utility tasks such as updating the positions and 
        pos_history attributes, checking boundary conditions, and recalculating 
        the current velocities and accelerations attributes.

        Parameters
        ----------
        dt : float
            length of time step to move all agents
        params : any, optional
            parameters to pass along to get_positions, if necessary
        ib_collisions : {None, 'default', 'sliding', 'sticky'}
            Boundary condition for immersed boundaries. If 'default', use the 
            default found in self.ib_condition. If None, turn off all 
            interaction with immersed boundaries. In sliding collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        update_time : bool, default=True
            whether or not to update the environment's time by dt. Probably 
            The only reason to change this to False is if there are multiple 
            swarm objects in the same environment - then you want to update 
            each before incrementing the time in the environment.
        silent : bool, default=False
            If True, suppress printing the updated time.

        See Also
        --------
        get_positions : 
            method that returns (but does not assign) the new positions of the 
            swarm after the time step dt, which Planktos users override in order 
            to specify their own, custom agent behavior.
        '''

        if ib_collisions == 'default':
            ib_collisions = self.ib_condition

        # Save current position to put in the history
        old_positions = self.positions.copy()
        old_velocities = self.velocities.copy()

        # Conditionally save props to put in the history too
        if self.props_history is not None:
            old_props = self.props.copy()

        # Check that something is left in the domain to move, and move it.
        if not np.all(self.positions.mask):
            # Update positions, preserving mask
            self.positions[:,:] = self.get_positions(dt, params)

        # Update history
        self.pos_history.append(old_positions)
        self.vel_history.append(old_velocities)
        if self.props_history is not None:
            self.props_history.append(old_props)
        
        # Update velocity and acceleration of swarm
        self.velocities[:,:] = (self.positions - old_positions)/dt
        self.accelerations[:,:] = (self.velocities - old_velocities)/dt

        # Apply boundary conditions (if anything was moving)
        if not np.all(self.positions.mask):
            self.apply_boundary_conditions(dt, ib_collisions=ib_collisions)
            self.after_move(dt, params)

        # Record new time
        if update_time:
            self.envir.time_history.append(self.envir.time)
            self.envir.time += dt
            if not silent:
                print('time = {}'.format(np.round(self.envir.time,11)))

            # Check for other swarms in environment and freeze them
            warned = False
            for s in self.envir.swarms:
                if s is not self and len(s.pos_history) < len(self.pos_history):
                    s.pos_history.append(s.positions.copy())
                    if not warned:
                        warnings.warn("Other swarms in the environment were not"+
                                      " moved during this environmental timestep.\n"+
                                      "If this was not your intent, call"+
                                      " envir.move_swarms instead of this method"+
                                      " to move all the swarms at once.",
                                      UserWarning)
                        warned = True



    def get_positions(self, dt, params=None):
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
        params : any, optional
            any other parameters necessary

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
        want to accidently edit self.positions directly, so make sure that you 
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
            given an agent/swarm property name, return the value(s). When 
            accessing a property in swarm.props, this can be preferred over 
            accessing the property directly through the because instead of 
            returning a pandas Series object (for a column in the DataFrame), it 
            automatically converts to a numpy array first.
        add_prop : add a new agent/swarm property or overwrite an old one
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



    def after_move(self, dt, params=None):
        '''This method is called after the swarm's spatial positions have been 
        updated via get_positions, but before the environment time has been 
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
        params : any, optional
            any other parameters necessary
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
            a column in the swarm.props DataFrame.
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

        return self.envir.interpolate_flow(positions, self.envir.dudt(time=time), 
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

        TIME_DEP = len(self.envir.flow[0].shape) != len(self.envir.L)
        flow_grad = None

        # If available, use the already calculuated gradient (if it's at the
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

        # Calculate if necessary, otherwise use cached copy
        if self.envir.DuDt is None or self.envir.DuDt_time != time:
            self.envir.calculate_DuDt(time=time)            

        # Interpolate at agent positions and return
        return self.envir.interpolate_flow(positions, self.envir.DuDt, 
                                           method='linear')



    def apply_boundary_conditions(self, dt, ib_collisions='sliding'):
        '''Apply boundary conditions to self.positions.
        
        There is no reason for a user to call this method directly; it is 
        automatically called by self.move after updating agent positions 
        according to the algorithm found in self.get_positions.

        This method compares current agent positions (self.positions) to the
        previous agent positions (last entry in self.pos_history) in order to
        first: determine if the agent collided with any immersed structures and
        if so, to update self.positions using a sliding collision algorithm 
        based on vector projection and second: assess whether or not any agents 
        exited the domain and if so, update their positions based on the 
        boundary conditions as specified in the enviornment class (self.envir).

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

            if not np.all(self.positions.mask) and self.envir.ibmesh.ndim == 4:
                # Get moving mesh info to pass into IBC routine
                start_mesh, end_mesh, max_meshpt_dist, max_mov = \
                    self._get_moving_mesh_info(dt)

            # if there are any masked agents, skip them in the loop
            if np.any(self.positions.mask):
                for n, startpt, endpt in \
                    zip(np.arange(self.N)[~ma.getmaskarray(self.positions[:,0])],
                        self.pos_history[-1][~ma.getmaskarray(self.positions[:,0]),:].copy(),
                        self.positions[~ma.getmaskarray(self.positions[:,0]),:].copy()
                        ):
                    if self.envir.ibmesh.ndim == 3:
                        self._IBC_routine_static(n, dt, startpt, endpt, ib_collisions)
                    else:
                        self._IBC_routine_moving(n, dt, startpt, endpt, start_mesh, 
                                                 end_mesh, max_meshpt_dist, 
                                                 max_mov, ib_collisions)
            # if all are masked, skip all boundary checks
            elif np.all(self.positions.mask):
                return
            # no masked agents: go through all of them
            else:
                for n in range(self.N):
                    startpt = self.pos_history[-1][n,:].copy()
                    endpt = self.positions[n,:].copy()
                    if self.envir.ibmesh.ndim == 3:
                        self._IBC_routine_static(n, dt, startpt, endpt, ib_collisions)
                    else:
                        self._IBC_routine_moving(n, dt, startpt, endpt, start_mesh, 
                                                 end_mesh, max_meshpt_dist, 
                                                 max_mov, ib_collisions)

        ##### Environment Boundary Conditions #####
        self._domain_BC_loop(dt, ib_collisions=ib_collisions)



    def _get_moving_mesh_info(self, dt):
        '''Helper function that returns interpolated mesh information needed for
        the immersed boundary interaction check.

        Parameters
        ----------
        dt : float
            size of the time step.

        Returns
        -------
        start_mesh : ndarray
        end_mesh : ndarray
        max_meshpt_dist : float
        max_mov : float
        '''

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
        
        return (start_mesh, end_mesh, max_meshpt_dist, max_mov)



    def _IBC_routine_static(self, idx, dt, startpt, endpt, ib_collisions='sliding'):
        '''Routine for checking static IB
        
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

        new_loc, dx = self._apply_internal_static_BC(startpt, endpt, 
                    self.envir.ibmesh, self.envir.max_meshpt_dist,
                    ib_collisions=ib_collisions)
        self.positions[idx] = new_loc
        if dx is not None:
            self.accelerations[idx] = (dx/dt - self.velocities[idx])/dt
            self.velocities[idx] = dx/dt
            self.ib_collision[idx] = True
        else:
            self.ib_collision[idx] = False



    def _IBC_routine_moving(self, idx, dt, startpt, endpt, start_mesh, end_mesh, 
                            max_meshpt_dist, max_mov, ib_collisions='sliding'):
        '''Routine for checking moving IB
        
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
        start_mesh : Nx2x2 or Nx3x3 ndarray
            starting position for the IB mesh
        end_mesh : Nx2x2 or Nx3x3 ndarray
            ending position for the IB mesh
        max_meshpt_dist : float
            maximum distance between two mesh vertices at either time
        max_mov : float
            maximum distance any mesh vertex moved
        dt : float
            Size of the time step.
        ib_collisions : {None, 'sliding' (default), 'sticky'}
            Type of interaction with immersed boundaries. If None, turn off all 
            interaction with immersed boundaries. In sliding collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        '''

        new_loc, dx = self._apply_internal_moving_BC(startpt, endpt, start_mesh, 
                    end_mesh, max_meshpt_dist, max_mov, dt,
                    ib_collisions=ib_collisions)
        self.positions[idx] = new_loc
        if dx is not None:
            self.accelerations[idx] = (dx/dt - self.velocities[idx])/dt
            self.velocities[idx] = dx/dt
            self.ib_collision[idx] = True
        else:
            self.ib_collision[idx] = False



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
            # figure out which exit crossing occured first and treat that as the 
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
                        if self.envir.ibmesh.ndim == 4:
                            start_mesh, end_mesh, max_meshpt_dist, max_mov = \
                                self._get_moving_mesh_info(dt)
                        for n, idx in enumerate(left_idx):
                            startpt = startpts[n]
                            endpt = self.positions[idx,:].copy()
                            if self.envir.ibmesh.ndim == 3:
                                self._IBC_routine_static(idx, dt, startpt, endpt, 
                                                         'sticky')
                            else:
                                self._IBC_routine_moving(idx, dt, startpt, endpt, 
                                                         start_mesh, end_mesh,
                                                         max_meshpt_dist, max_mov,
                                                         'sticky')
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
                        if self.envir.ibmesh.ndim == 4:
                            start_mesh, end_mesh, max_meshpt_dist, max_mov = \
                                self._get_moving_mesh_info(dt)
                        for n, idx in enumerate(left_idx):
                            startpt = self.positions[idx,:] - \
                                s_array[n]*self.velocities[idx,:]
                            endpt = self.positions[idx,:].copy()
                            if self.envir.ibmesh.ndim == 3:
                                self._IBC_routine_static(idx, dt, startpt, 
                                                         endpt, ib_collisions)
                            else:
                                self._IBC_routine_moving(idx, dt, startpt, endpt, 
                                                         start_mesh, end_mesh,
                                                         max_meshpt_dist, max_mov,
                                                         ib_collisions)
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
                        if self.envir.ibmesh.ndim == 4:
                            start_mesh, end_mesh, max_meshpt_dist, max_mov = \
                                self._get_moving_mesh_info(dt)
                        for n, idx in enumerate(right_idx):
                            startpt = startpts[n]
                            endpt = self.positions[idx,:].copy()
                            if self.envir.ibmesh.ndim == 3:
                                self._IBC_routine_static(idx, dt, startpt, 
                                                         endpt, 'sticky')
                            else:
                                self._IBC_routine_moving(idx, dt, startpt, endpt, 
                                                         start_mesh, end_mesh,
                                                         max_meshpt_dist, max_mov,
                                                         'sticky')
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
                        if self.envir.ibmesh.ndim == 4:
                            start_mesh, end_mesh, max_meshpt_dist, max_mov = \
                                self._get_moving_mesh_info(dt)
                        for n, idx in enumerate(right_idx):
                            startpt = self.positions[idx,:] - \
                                s_array[n]*self.velocities[idx,:]
                            endpt = self.positions[idx,:].copy()
                            if self.envir.ibmesh.ndim == 3:
                                self._IBC_routine_static(idx, dt, startpt, 
                                                         endpt, ib_collisions)
                            else:
                                self._IBC_routine_moving(idx, dt, startpt, endpt, 
                                                         start_mesh, end_mesh,
                                                         max_meshpt_dist, max_mov,
                                                         ib_collisions)
                    # further domain crossings are possible. if this happens, 
                    #   velocity should be the same as original velocity b/c 
                    #   immersed boundaries do not intersect with domain bndry,
                    #   so agent has slid off with original velocity heading.
                else:
                    raise NameError

        ##### All BC applied to first exit. Conduct recursion if necessary #####
        if mult_idx is not None:
            self._domain_BC_loop(dt, ib_collisions, idx_array=mult_idx)


    
    @staticmethod
    def _apply_internal_static_BC(startpt, endpt, mesh, max_meshpt_dist, 
                                  old_intersection=None, kill=False, 
                                  ib_collisions='sliding'):
        '''Apply internal boundaries to a trajectory starting and ending at
        startpt and endpt, returning a new endpt (or the original one) as
        appropriate.

        Parameters
        ----------
        startpt : length 2 or 3 array
            start location for agent trajectory
        endpt : length 2 or 3 array
            end location for agent trajectory
        mesh : Nx2x2 or Nx3x3 array 
            eligible mesh elements to check for intersection
        max_meshpt_dist : float
            max distance between two points on a mesh element
            (used to determine how far away from startpt to search for
            mesh elements)
        old_intersection : list-like of data
            (for internal use only) records the last intersection in the 
            recursion to check if we are bouncing back and forth between two 
            boundaries as a result of a concave angle and the right kind of 
            trajectory vector.
        kill : bool
            (for internal use only) set to True in 3D case if we have previously 
            slid along the boundary line between two mesh elements. This 
            prevents such a thing from happening more than once, in case of 
            pathological cases.
        ib_collisions : {'sliding' (default), 'sticky'}
            Type of interaction with immersed boundaries. In sliding 
            collisions, conduct recursive vector projection until the length of
            the original vector is exhausted. In sticky collisions, just return 
            the point of intersection.

        Returns
        -------
        newendpt : length 2 or 3 array
            new end location for agent trajectory
        dx : length 2 or 3 array, or None
            change in position for agent after IB collision - based on first
            collision point and final location. If no IB collision, None.

        Acknowledgements
        ---------------
        Appreciation goes to Anne Ho, for pointing out that the centroid of an
        equalateral triangle is further away from any of its vertices than I had 
        originally assumed it was.
        '''

        if len(startpt) == 2:
            DIM = 2
        else:
            DIM = 3

        # We only want to check mesh elements that could feasibly intersect 
        #   the line segment startpt endpt. Mesh elements are identified by 
        #   their vertices; the question is: how far away from the line segment 
        #   do we need to look for mesh vertices? Part of the answer to this 
        #   question depends on how big mesh elements can be, since the longer 
        #   they are, the further away their vertices could be.

        # In barycentric coordinates, the centoid is always (1/3,1/3,1/3), and
        #   the entries of the coordinates must add to 1. This suggests that the
        #   furthest away you can be from every vertex simultaneously is 2/3 
        #   down the medians (an increase in one barycentric coordinate results 
        #   in a decrease in the others). Since the median length is bounded 
        #   above by the length of the longest side of the triangle, circles 
        #   centered at each vertex that are 2/3 times the length of the longest 
        #   triangle side should be sufficient to cover any triangle.
        # More precisely: equalateral triangles are probably the worst case. If 
        #   so, all medians have length l*sqrt(3/4), where l is the length of a 
        #   side of the triangle. This implies circles of radius l*sqrt(3/4)*2/3
        #   are a strict lower bound on covering any circle.

        # The result of this argument, from the worst-case scenario equalateral 
        #   triangle in our collection, forms the radius we need to search from 
        #   the line segment of travel in order to find vertices of mesh elements
        #   that we potentially intersected.
        search_rad = max_meshpt_dist*2/3

        # Find all mesh elements that have vertex points within search_rad of 
        #   the trajectory segment.
        close_mesh = swarm._get_eligible_static_mesh_elements(startpt, endpt, mesh, 
                                                              search_rad)

        # Get intersections
        if DIM == 2:
            intersection = geom.seg_intersect_2D(startpt, endpt,
                close_mesh[:,0,:], close_mesh[:,1,:])
        else:
            intersection = geom.seg_intersect_3D_triangles(startpt, endpt,
                close_mesh[:,0,:], close_mesh[:,1,:], close_mesh[:,2,:])

        # Return endpt we already have if None.
        if intersection is None:
            return endpt, None
        
        # If we do have an intersection:
        if ib_collisions == 'sliding':
            # Project remaining piece of vector onto mesh and repeat processes 
            #   as necessary until we have a final result.
            new_pos = swarm._project_and_slide(startpt, endpt, intersection, mesh,
                                               close_mesh, max_meshpt_dist, DIM,
                                               old_intersection, kill)
            return new_pos, new_pos - intersection[0]
        
        elif ib_collisions == 'sticky':
            # Return the point of intersection
            
            # small number to perturb off of the actual boundary in order to avoid
            #   roundoff errors that would allow penetration
            EPS = 1e-7

            back_vec = (startpt-endpt)/np.linalg.norm(endpt-startpt)
            return intersection[0] + back_vec*EPS, np.zeros((DIM))



    @staticmethod
    def _get_eligible_static_mesh_elements(startpt, endpt, mesh, search_rad):
        '''
        From a list of mesh elements (mesh), find all elements that have vertex 
        points within search_rad of the trajectory segment startpt,endpt.
        '''
        
        pt_bool = geom.closest_dist_btwn_line_and_pts(startpt, endpt, 
            mesh.reshape((mesh.shape[0]*mesh.shape[1],mesh.shape[2])))<=search_rad
        pt_bool = pt_bool.reshape((mesh.shape[0],mesh.shape[1]))
        return mesh[np.any(pt_bool,axis=1)]
    


    @staticmethod
    def _apply_internal_moving_BC(startpt, endpt, start_mesh, end_mesh, 
                                  max_meshpt_dist, max_mov, dt, 
                                  old_intersection=None, kill=False, 
                                  ib_collisions='sliding'):
        '''Apply internal boundaries to a trajectory starting and ending at
        startpt and endpt, returning a new endpt (or the original one) as
        appropriate.

        Parameters
        ----------
        startpt : length 2 or 3 array
            start location for agent trajectory
        endpt : length 2 or 3 array
            end location for agent trajectory
        start_mesh : Nx2x2 or Nx3x3 array
            starting position for the mesh
        end_mesh : Nx2x2 or Nx3x3 array
            ending position for the mesh
        max_meshpt_dist : float
            maximum distance between two mesh vertices at either time
        max_mov : float
            maximum distance any mesh vertex moved
        dt : float
            size of the time step
        old_intersection : list-like of data
            (for internal use only) records the last intersection in the 
            recursion to check if we are bouncing back and forth between two 
            boundaries as a result of a concave angle and the right kind of 
            trajectory vector.
        kill : bool
            (for internal use only) set to True in 3D case if we have previously 
            slid along the boundary line between two mesh elements. This 
            prevents such a thing from happening more than once, in case of 
            pathological cases.
        ib_collisions : {'sliding' (default), 'sticky'}
            Type of interaction with immersed boundaries. In sliding 
            collisions, conduct recursive vector projection until the length of
            the original vector is exhausted. In sticky collisions, just return 
            the point of intersection.

        Returns
        -------
        newendpt : length 2 or 3 array
            new end location for agent trajectory
        dx : length 2 or 3 array, or None
            change in position for agent after IB collision - based on first
            collision point and final location. If no IB collision, None.
        '''

        if len(startpt) == 2:
            DIM = 2
        else:
            DIM = 3

        # See static case for derivation of the 2/3 argument
        search_rad = max_meshpt_dist*2/3

        close_mesh_start, close_mesh_end = \
            swarm._get_eligible_moving_mesh_elements(startpt, endpt, start_mesh, 
                                                     end_mesh, max_mov, search_rad)
        
        # Get intersections
        if DIM == 2:
            # find interesections between line segment of motion and 
            #   quadrilateral in 3D (t,x,y) space
            intersection = geom.seg_intersect_2D_multilinear_poly(startpt, endpt,
                                close_mesh_start[:,0,:], close_mesh_start[:,1,:],
                                close_mesh_end[:,0,:], close_mesh_end[:,1,:])
        else:
            # find interesections between line segment of motion and 
            #   hyper-quadrilateral (hyper-plane) in 4D (t,x,y,z) space
            raise NotImplementedError("3D moving meshes not currently supported.")
        
    
        # Return endpt we already have if None.
        if intersection is None:
            return endpt, None
        
        # If we have an intersection with this agent, apply boundary condition
        if ib_collisions == 'sticky':
            # Return the point of intersection
            
            # small number to perturb off of the actual boundary in order to avoid
            #   roundoff errors that would allow boundary penetration
            EPS = 1e-7

            if DIM == 2:
                x = intersection[0]
                Q0 = intersection[2]
                Q1 = intersection[3]
                idx = intersection[4]

                # Get the relative position of intersection within the mesh element
                # Use max in case the element is vertical or horizontal
                s = max((x[0]-Q0[0])/(Q1[0]-Q0[0]),(x[1]-Q0[1])/(Q1[1]-Q0[1]))
                if idx is None:
                    st_elem = close_mesh_start
                    dt_elem = close_mesh_end
                else:
                    st_elem = close_mesh_start[idx]
                    dt_elem = close_mesh_end[idx]
                # Translate to final position of mesh element in this time step
                new_pos = dt_elem[0,:] + s*(dt_elem[1,:] - dt_elem[0,:])
                
                # Perturb a small bit off of the boundary.
                #   This needs to be on the side of the element the motion 
                #   came from.

                # Find perpendicular direction based off of starting positions
                #   of both mesh element and agent
                perp_vec = np.array([st_elem[1,1]-st_elem[0,1],st_elem[0,0]-st_elem[1,0]])
                perp_vec /= np.linalg.norm(perp_vec)
                # find the side of the mesh element the agent started on
                to_pt_vec = startpt - st_elem[0,:]
                signum = np.dot(to_pt_vec,perp_vec)/np.linalg.norm(np.dot(to_pt_vec,perp_vec))

                # Now, perturb perpendicular from the end position of the element
                perp_vec = np.array([dt_elem[1,1]-dt_elem[0,1],dt_elem[0,0]-dt_elem[1,0]])
                perp_vec *= signum/np.linalg.norm(perp_vec)
                return new_pos + perp_vec*EPS, new_pos - x
            else:
                raise NotImplementedError("3D moving meshes not currently supported.")
        else:
            raise NotImplementedError("sliding moving meshes not currently supported.")



    @staticmethod
    def _get_eligible_moving_mesh_elements(startpt, endpt, start_mesh, end_mesh, 
                                           max_mov, search_rad):
        '''
        From starting and ending points for the mesh, find all elements that 
        have vertex points which passed within search_rad of the trajectory 
        segment startpt,endpt. Return a tuple of the start and end eligible meshes.
        '''

        #Unfortunately, finding the closest distance between two lines is likely 
        #   slower than just finding the distance between a line and a point as 
        #   in the static case. Instead, do a coarse rule-out and then refine 
        #   with closest distance between two lines.

        # 1/2 a distance between two points is the furthest away you can be and 
        #   still intersect the line segment between them
        outer_rad = search_rad + 0.5*np.linalg.norm(endpt-startpt) + 0.5*max_mov
        start_mesh_r = start_mesh.reshape((start_mesh.shape[0]*start_mesh.shape[1],
                                           start_mesh.shape[2]))
        end_mesh_r = end_mesh.reshape((end_mesh.shape[0]*end_mesh.shape[1],
                                       end_mesh.shape[2]))
        dist_array = np.empty((4, start_mesh_r.shape[0]))
        dist_array[0,:] = np.linalg.norm(startpt - start_mesh_r, axis=1) < outer_rad
        dist_array[1,:] = np.linalg.norm(startpt - end_mesh_r, axis=1) < outer_rad
        dist_array[2,:] = np.linalg.norm(endpt - start_mesh_r, axis=1) < outer_rad
        dist_array[3,:] = np.linalg.norm(endpt - end_mesh_r, axis=1) < outer_rad
        outer_bool = np.any(dist_array, axis=0)

        # anything within the outer radius gets a better check
        dist_list = geom.closest_dist_btwn_two_lines(startpt, endpt,
            start_mesh_r[outer_bool,:], end_mesh_r[outer_bool,:])
        inner_bool = dist_list < search_rad

        # refine outer_bool with inner_bool
        outer_bool[outer_bool] = inner_bool

        pt_bool = outer_bool.reshape((start_mesh.shape[0], start_mesh.shape[1]))
        return (start_mesh[np.any(pt_bool,axis=1)], end_mesh[np.any(pt_bool,axis=1)])



    @staticmethod
    def _project_and_slide(startpt, endpt, intersection, mesh, close_mesh, 
                           max_meshpt_dist, DIM, old_intersection=None, 
                           kill=False):
        '''Once we have an intersection point with an immersed mesh, project and 
        slide the agent along the mesh for its remaining movement, and determine 
        what happens if we fall off the edge of the element in all angle cases 
        (e.g. some kind of recursion of detecting further intersections and 
        resulting in additional vector projection)

        This, along with the intersection detection routines, is the real 
        workhorse of immersed mesh interaction!

        Parameters
        ----------
        startpt : length 2 or 3 array
            original start point of movement, before intersection
        endpt : length 2 or 3 array
            original end point of movement, w/o intersection
        intersection : list-like of data
            result of seg_intersect_2D or seg_intersect_3D_triangles. various 
            information about the intersection with the immersed mesh element
        mesh : Nx2x2 or Nx3x3 array
            full immersed boundary mesh (for recalculating close_mesh)
        close_mesh : Nx2x2 or Nx3x3 array 
            eligible (nearby) mesh elements for interaction
        max_meshpt_dist : float
            max distance between two points on a mesh element (used to determine 
            how far away from startpt to search for mesh elements). Used here 
            for possible recursion
        DIM : int
            dimension of system, either 2 or 3
        old_intersection : list-like of data
            (for internal use only) records the last intersection in the 
            recursion to check if we are bouncing back and forth between two 
            boundaries as a result of a concave angle and the right kind of 
            trajectory vector.
        kill : bool
            (for internal use only) set to True in 3D case if we have previously 
            slid along the boundary line between two mesh elements. This 
            prevents such a thing from happening more than once, in case of 
            pathological cases.

        Returns
        -------
        newendpt : length 2 or 3 array
            new endpoint for movement after projection
        '''

        # small number to perturb off of the actual boundary in order to avoid
        #   roundoff errors that would allow penetration
        EPS = 1e-7

        # Project remaining piece of vector from intersection onto mesh and get 
        #   a unit normal pointing out from the simplex
        # The first two entries of an intersection object are always:
        #   0) The point of intersection
        #   1) The fraction of line segment traveled before intersection occurred
        # NOTE: In the case of 2D, intersection[2] is a unit vector on the line
        #       In the case of 3D, intersection[2] is a unit normal to the triangle
        #   The remaining entries give the vertices of the line/triangle

        # Get leftover portion of travel vector
        vec = (1-intersection[1])*(endpt-startpt)
        if DIM == 2:
            # project vec onto the line
            proj = np.dot(vec,intersection[2])*intersection[2]
            # vec - proj is a normal that points from outside bndry inside
            #   reverse direction and normalize to get unit vector pointing out
            norm_out_u = (proj-vec)/np.linalg.norm(proj-vec)
        if DIM == 3:
            # get a unit normal vector pointing back the way we came
            norm_out_u = -np.sign(np.dot(vec,intersection[2]))*intersection[2]
            # Get the component of vec that lies in the plane. We do this by
            #   subtracting off the component which is in the normal direction
            proj = vec - np.dot(vec,norm_out_u)*norm_out_u

        # get a unit vector of proj for adding EPS in proj direction
        if np.linalg.norm(proj) > 0:
            proj_u = proj/np.linalg.norm(proj)
        else:
            proj_u = 0
            
        # IMPORTANT: there can be roundoff error, so proj should be considered
        #   an approximation to an in-boundary slide.
        # For this reason, pull newendpt back a bit for numerical stability
        newendpt = intersection[0] + proj + EPS*norm_out_u

        ################################
        ###########    2D    ###########
        ################################
        if DIM == 2:
            # Detect sliding off 1D edge
            # Equivalent to going past the endpoints
            mesh_el_len = np.linalg.norm(intersection[4] - intersection[3])
            Q0_dist = np.linalg.norm(newendpt-EPS*norm_out_u - intersection[3])
            Q1_dist = np.linalg.norm(newendpt-EPS*norm_out_u - intersection[4])
            # Since we are sliding on the mesh element, if the distance from
            #   our new location to either of the mesh endpoints is greater
            #   than the length of the mesh element, we must have gone beyond
            #   the segment.
            if Q0_dist > mesh_el_len+EPS or Q1_dist > mesh_el_len+EPS:
                ##########      Went past either Q0 or Q1      ##########

                # We check assuming the agent slid an additional EPS, because
                #   if we end up in the non-concave case and fall off, we will
                #   want to go EPS further so as to avoid corner penetration
                #   The result of adding EPS in both the projection and normal
                #   directions (below) is that some concave intersections will be
                #   discovered (all acute, some obtuse) but others won't. However,
                #   All undiscovered ones will be obtuse, so while re-detecting
                #   the subsequent intersection is not the most efficient thing,
                #   it should be stable.

                # check for new crossing of segment attached to the
                #   current line segment
                # First, find adjacent mesh segments. These are ones that share
                #   one of, but not both, endpoints with the current segment.
                #   This will also pick up our current segment, but we shouldn't
                #   be intersecting it. And if by some numerical error we do,
                #   we need to treat it.
                pt_bool = np.logical_or(
                    np.isclose(np.linalg.norm(close_mesh.reshape(
                    (close_mesh.shape[0]*close_mesh.shape[1],close_mesh.shape[2]))-intersection[3],
                    axis=1),0),
                    np.isclose(np.linalg.norm(close_mesh.reshape(
                    (close_mesh.shape[0]*close_mesh.shape[1],close_mesh.shape[2]))-intersection[4],
                    axis=1),0)
                )
                pt_bool = pt_bool.reshape((close_mesh.shape[0],close_mesh.shape[1]))
                adj_mesh = close_mesh[np.any(pt_bool,axis=1)]
                # Check for intersection with these segemtns, but translate 
                #   start/end points back from the segment a bit for numerical stability
                # This has already been done for newendpt
                # Also go EPS further than newendpt for stability for what follows
                if len(adj_mesh) > 0:
                    adj_intersect = geom.seg_intersect_2D(intersection[0]+EPS*norm_out_u,
                        newendpt+EPS*proj_u, adj_mesh[:,0,:], adj_mesh[:,1,:])
                else:
                    adj_intersect = None
                if adj_intersect is not None:
                    ##########  Went past and intersected adjoining element! ##########
                    # check that we haven't intersected this before
                    #   (should be impossible)
                    if old_intersection is not None and\
                        np.all(adj_intersect[3] == old_intersection[3]) and\
                        np.all(adj_intersect[4] == old_intersection[4]):
                        # Going back and forth! Movement stops here.
                        # NOTE: This happens when 1) trying to go through a
                        #   mesh element, you 2) slide and intersect another
                        #   mesh element. The angle between elements is acute,
                        #   so you 3) slide back into the same element you
                        #   intersected in (1). 
                        # Back away from the intersection point slightly for
                        #   numerical stability and stay put.
                        return adj_intersect[0] - EPS*proj_u
                    # Otherwise:
                    if Q0_dist > Q1_dist:
                        # went past Q1; base vectors off Q1 vertex
                        idx = 4
                        nidx = 3
                    else:
                        # went past Q0; base vectors off Q0 vertex
                        idx = 3
                        nidx = 4
                    vec0 = intersection[nidx] - intersection[idx]
                    adj_idx = np.argmin([
                        np.linalg.norm(adj_intersect[3]-intersection[idx]),
                        np.linalg.norm(adj_intersect[4]-intersection[idx])]) + 3
                    vec1 = adj_intersect[adj_idx] - intersection[idx]
                    # Determine if the angle of mesh elements is acute or obtuse.
                    if np.dot(vec0,vec1) >= 0:
                        # Acute. Movement stops here.
                        # Back away from the intersection point slightly for
                        #   numerical stability and stay put
                        return adj_intersect[0] - EPS*proj_u
                    else:
                        # Obtuse. Repeat project_and_slide on new segment,
                        #   but send along info about old segment so we don't
                        #   get in an infinite loop.
                        # Also, regenerate eligible mesh elements based on the 
                        #   new location.
                        search_rad = max_meshpt_dist*2/3
                        close_mesh = swarm._get_eligible_static_mesh_elements(
                            intersection[0]+EPS*norm_out_u, newendpt+EPS*proj_u, 
                            mesh, search_rad)

                        return swarm._project_and_slide(intersection[0]+EPS*norm_out_u, 
                                                        newendpt+EPS*proj_u, 
                                                        adj_intersect, mesh, 
                                                        close_mesh, max_meshpt_dist, 
                                                        DIM, intersection)

                ######  Went past, but did not intersect adjoining element! ######

                # NOTE: There could still be an adjoining element at >180 degrees
                #   But we will project along original heading as if there isn't one
                #   and subsequently detect any intersections.
                # DIM == 2, adj_intersect is None
                
                if Q0_dist > Q1_dist:
                    ##### went past Q1 #####
                    # put a new start point at the point crossing+EPS and bring out
                    #   EPS*norm_out_u
                    newstartpt = intersection[4] + EPS*proj_u + EPS*norm_out_u
                elif Q0_dist < Q1_dist:
                    ##### went past Q0 #####
                    newstartpt = intersection[3] + EPS*proj_u + EPS*norm_out_u
                else:
                    raise RuntimeError("Impossible case??")
                
                # continue full amount of remaining movement along original heading
                #   starting at newstartpt
                orig_unit_vec = (endpt-startpt)/np.linalg.norm(endpt-startpt)
                newendpt = newstartpt + np.linalg.norm(newendpt-newstartpt)*orig_unit_vec
                # repeat process to look for additional intersections
                #   pass along current intersection in case of obtuse concave case
                new_loc, dx = swarm._apply_internal_static_BC(newstartpt, newendpt,
                                                              mesh, max_meshpt_dist,
                                                              intersection)
                return new_loc

            else:
                ##########      Did not go past either Q0 or Q1      ##########
                # We simply end on the mesh element
                return newendpt
        
        ################################
        ###########    3D    ###########
        ################################
        if DIM == 3:
            # Detect sliding off 2D edge using seg_intersect_2D
            Q0_list = np.array(intersection[3:])
            Q1_list = Q0_list[(1,2,0),:]
            # go a little further along trajectory to treat acute case
            tri_intersect = geom.seg_intersect_2D(intersection[0] + EPS*norm_out_u,
                                                    newendpt + EPS*proj_u,
                                                    Q0_list, Q1_list)
            # if we reach the triangle boundary, check for a acute crossing
            #   then project overshoot onto original heading and add to 
            #   intersection point
            if tri_intersect is not None:
                ##########      Went past triangle boundary      ##########
                # check for new crossing of simplex attached to the current triangle
                # First, find adjacent mesh segments. These are ones that share
                #   one of, but not both, endpoints with the current segment.
                #   This will also pick up our current segment, but we shouldn't
                #   be intersecting it. And if by some numerical error we do,
                #   we need to treat it.
                pt_bool = np.logical_or(np.logical_or(
                    np.isclose(np.linalg.norm(close_mesh.reshape(
                    (close_mesh.shape[0]*close_mesh.shape[1],close_mesh.shape[2]))-intersection[3],
                    axis=1),0),
                    np.isclose(np.linalg.norm(close_mesh.reshape(
                    (close_mesh.shape[0]*close_mesh.shape[1],close_mesh.shape[2]))-intersection[4],
                    axis=1),0)),
                    np.isclose(np.linalg.norm(close_mesh.reshape(
                    (close_mesh.shape[0]*close_mesh.shape[1],close_mesh.shape[2]))-intersection[5],
                    axis=1),0)
                )
                pt_bool = pt_bool.reshape((close_mesh.shape[0],close_mesh.shape[1]))
                adj_mesh = close_mesh[np.any(pt_bool,axis=1)]
                # check for intersection, but translate start/end points back
                #   from the simplex a bit for numerical stability
                # this has already been done for newendpt
                # also go EPS further than newendpt for stability for what follows
                if len(adj_mesh) > 0:
                    adj_intersect = geom.seg_intersect_3D_triangles(
                        intersection[0]+EPS*norm_out_u,
                        newendpt+EPS*proj_u, adj_mesh[:,0,:],
                        adj_mesh[:,1,:], adj_mesh[:,2,:])
                else:
                    adj_intersect = None
                if adj_intersect is not None:
                    ##########      Intersected adjoining element!      ##########
                    # Acute or slightly obtuse intersection. In 3D, we will slide
                    #   regardless of if it is acute or not.
                    # First, detect if we're hitting a simplex we've already 
                    #   been on before. If so, we follow the intersection line.
                    if old_intersection is not None and\
                        np.all(adj_intersect[3] == old_intersection[3]) and\
                        np.all(adj_intersect[4] == old_intersection[4]) and\
                        np.all(adj_intersect[5] == old_intersection[5]):
                        # Going back and forth between two elements. Project
                        #   onto the line of intersection.
                        # NOTE: This happens when 1) trying to go through a
                        #   mesh element, you 2) slide and intersect another
                        #   mesh element. The angle between elements is acute,
                        #   so you 3) slide back into the same element you
                        #   intersected in (1).
                        # NOTE: In rare cases, solving this problem by following
                        #   the line of intersection can result in an infinite
                        #   loop. For example, when going toward the point of
                        #   a tetrahedron. This is what the kill switch is for.
                        if kill:
                            # End here to prevent possible bad behavior.
                            return adj_intersect[0] - EPS*proj_u

                        kill = True
                        # Find the points in common between adj_intersect and
                        #   intersection
                        
                        # Enter debugging here to test algorithm
                        # import pdb; pdb.set_trace()
                        dist_mat = distance.cdist(intersection[3:],adj_intersect[3:])
                        pt_bool = np.isclose(dist_mat, np.zeros_like(dist_mat),
                                             atol=1e-6).any(1)
                        two_pts = np.array(intersection[3:])[pt_bool]
                        assert two_pts.shape[0] == 2, "Other than two points found?"
                        assert two_pts.shape[1] == 3, "Points aren't 3D?"
                        # Get a unit vector from these points
                        vec_intersect = two_pts[1,:] - two_pts[0,:]
                        vec_intersect /= np.linalg.norm(vec_intersect)
                        # Back up a tad from the intersection point, and project
                        #   leftover movement in the direction of intersection vector
                        newstartpt = adj_intersect[0] - EPS*proj_u
                        # Get leftover portion of the travel vector
                        vec_to_project = (1-adj_intersect[1])*proj
                        # project vec onto the line
                        proj_vec = np.dot(vec_to_project,vec_intersect)*vec_intersect
                        # Get new endpoint
                        newendpt = newstartpt + proj_vec
                        # Check for more intersections
                        new_loc, dx = swarm._apply_internal_static_BC(newstartpt, newendpt,
                                                               mesh, max_meshpt_dist,
                                                               adj_intersect, kill)
                        return new_loc

                    # Not an already discovered mesh element.
                    # We slide. Pass along info about the old element and 
                    #   regenerate eligible mesh segments.
                    search_rad = max_meshpt_dist*2/3
                    close_mesh = swarm._get_eligible_static_mesh_elements(
                        intersection[0]+EPS*norm_out_u, 
                        newendpt+EPS*proj_u+EPS*norm_out_u, 
                        mesh, search_rad)
                    return swarm._project_and_slide(intersection[0]+EPS*norm_out_u, 
                                                    newendpt+EPS*proj_u+EPS*norm_out_u, 
                                                    adj_intersect, mesh, close_mesh,
                                                    max_meshpt_dist, DIM,
                                                    intersection, kill)

                ##########      Did not intersect adjoining element!      ##########
                # NOTE: There could still be an adjoining element w/ >180 connection
                #   But we will project along original heading as if there isn't
                #   one and subsequently detect any intersections.
                # DIM == 3, adj_intersect is None
                # put the new start point on the edge of the simplex (+EPS) and
                #   continue full amount of remaining movement along original heading
                newstartpt = tri_intersect[0] + EPS*proj_u + EPS*norm_out_u
                orig_unit_vec = (endpt-startpt)/np.linalg.norm(endpt-startpt)
                # norm(newendpt - tri_intersect[0]) is the length of the overshoot.
                newendpt = newstartpt + np.linalg.norm(newendpt-tri_intersect[0])*orig_unit_vec
                # repeat process to look for additional intersections
                new_loc, dx = swarm._apply_internal_static_BC(newstartpt, newendpt, 
                                                              mesh, max_meshpt_dist,
                                                              intersection, kill)
                return new_loc
            else:
                # otherwise, we end on the mesh element
                return newendpt



    def _calc_basic_stats(self, DIM3, t_indx=None):
        ''' Return basic stats about % agents remaining, fluid velocity, and 
        agent velocity for plot printing.

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
        avg_spd : float
            average fluid speed
        max_spd : float
            maximum fluid speed
        avg_spd_x : float
            average x-component of fluid velocity
        avg_spd_y : float
            average y-component of fluid velocity
        avg_spd_z : float, 3D only
            average z-component of fluid velocity
        avg_swrm_vel : array
            average agent velocity
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

        # get average swarm velocity
        if t_indx is None:
            if np.all(self.velocities.mask):
                vel_data = np.zeros(self.velocities.shape)
            elif np.any(self.velocities.mask):
                vel_data = self.velocities[~self.velocities.mask.any(axis=1)]
            else:
                vel_data = self.velocities
            avg_swrm_vel = vel_data.mean(axis=0)
        elif t_indx == 0:
            avg_swrm_vel = np.zeros(len(self.envir.L))
        else:
            vel_data = (self.pos_history[t_indx] - self.pos_history[t_indx-1])/(
                        self.envir.time_history[t_indx]-self.envir.time_history[t_indx-1])
            avg_swrm_vel = vel_data.mean(axis=0)

        if self.envir.flow is None and not DIM3:
            return perc_left, 0, 0, 0, 0, avg_swrm_vel
        elif self.envir.flow is None and DIM3:
            return perc_left, 0, 0, 0, 0, 0, avg_swrm_vel

        if not DIM3:
            # 2D flow
            # get current fluid flow info
            if len(self.envir.flow[0].shape) == 2:
                # temporally constant flow
                flow = self.envir.flow
            else:
                # temporally changing flow
                flow = self.envir.interpolate_temporal_flow(t_index=t_indx)
            flow_spd = np.sqrt(flow[0]**2 + flow[1]**2)
            avg_spd_x = flow[0].mean()
            avg_spd_y = flow[1].mean()
            avg_spd = flow_spd.mean()
            max_spd = flow_spd.max()
            return perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_swrm_vel

        else:
            # 3D flow
            if len(self.envir.flow[0].shape) == 3:
                # temporally constant flow
                flow = self.envir.flow
            else:
                # temporally changing flow
                flow = self.envir.interpolate_temporal_flow(t_indx)
            flow_spd = np.sqrt(flow[0]**2 + flow[1]**2 + flow[2]**2)
            avg_spd_x = flow[0].mean()
            avg_spd_y = flow[1].mean()
            avg_spd_z = flow[2].mean()
            avg_spd = flow_spd.mean()
            max_spd = flow_spd.max()
            return perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel



    def plot(self, t=None, filename=None, blocking=True, dist='density', 
             fluid=None, clip=None, figsize=None, circ_rad=0.25, plot_heading=True,
             save_kwargs=None, azim=None, elev=None):
        '''Plot the position of the swarm at time t, or at the current time
        if no time is supplied. The actual time plotted will depend on the
        history of movement steps; the closest entry in
        environment.time_history will be shown without interpolation.
        
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
                vort = self.envir.get_2D_vorticity(t_indx=loc)
                if clip is not None:
                    norm = colors.Normalize(-abs(clip),abs(clip),clip=True)
                else:
                    norm = None
                ax.pcolormesh(self.envir.flow_points[0], self.envir.flow_points[1], 
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
                M = int(round(len(self.envir.flow_points[0])/x_num))
                N = int(round(len(self.envir.flow_points[1])/y_num))
                # get worse case max velocity vector for scaling
                max_u = self.envir.flow[0].max(); max_v = self.envir.flow[1].max()
                max_mag = np.linalg.norm(np.array([max_u,max_v]))
                if len(self.envir.flow[0].shape) > 2:
                    flow = self.envir.interpolate_temporal_flow(t_index=loc)
                else:
                    flow = self.envir.flow
                ax.quiver(self.envir.flow_points[0][::M], self.envir.flow_points[1][::N],
                          flow[0][::M,::N].T, flow[1][::M,::N].T, 
                          scale=max_mag*5, alpha=0.2)

            # ibmesh (if moving and not a current time - otherwise, done already)
            if mesh_col is not None and self.envir.ibmesh.ndim == 4 and t is not None:
                ibmesh = self.interpolate_temporal_mesh(time=t)
                mesh_col.set_segments(ibmesh)

            # Create marker headings to add to scatter
            paths = []
            circle = Path.circle(radius=circ_rad)
            if plot_heading:
                line_codes = np.array([Path.MOVETO, Path.LINETO])
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
                        paths.append(Path(verts, codes))
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
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_swrm_vel = \
                self._calc_basic_stats(DIM3=False, t_indx=loc)
            plt.figtext(0.77, 0.77,
                        '{:.1f}% remain\n'.format(perc_left)+
                        '\n  ------ Info ------\n'+
                        r'Fluid $v_{max}$'+': {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                        r'Fluid $\overline{v}$'+': {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                        r'Agent $\overline{v}$'+': {:.1g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units),
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
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel = \
                self._calc_basic_stats(DIM3=True, t_indx=loc)
            ax.text2D(0.75, 0.9, r'Fluid $v_{max}$'+': {:.2g} {}/s\n'.format(max_spd, self.envir.units)+
                      r'Fluid $v_{avg}$'+': {:.2g} {}/s\n'.format(avg_spd, self.envir.units)+
                      r'Agent $v_{avg}$'+': {:.2g} {}/s'.format(np.linalg.norm(avg_swrm_vel), self.envir.units),
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



    def plot_all(self, movie_filename=None, frames=None, downsamp=None, fps=10, 
                 dist='density', fluid=None, clip=None, figsize=None, circ_rad=0.25,
                 plot_heading=True, save_kwargs=None, writer_kwargs=None, 
                 azim=None, elev=None):
        ''' Plot the history of the swarm's movement, incl. current time in 
        successively updating plots or saved as a movie file. A movie file is
        created if movie_filename is specified.

        Agent colors will be read from the 'color' column of props if it exists; 
        otherwise it will default to the color attribute of the swarm.
        
        Parameters
        ----------
        movie_filename : string, optional
            file name to save movie as. file extension will determine the type
            of file saved.
        frames : iterable of integers, optional. 
            If None, plot the entire history of the swarm's movement including 
            the present time, with each step being a frame in the animation. If 
            an iterable, plot only the time steps of the swarm as indexed by
            the iterable (note, this is an interable of the time step indices, 
            not the time in seconds at those time steps!).
        downsamp : iterable of int or int, optional 
            If None, do not downsample the agents - plot them all. If an integer, 
            plot only the first n agents (equivalent to range(downsamp)). 
            If an iterable, plot only the agents specified. In all cases, 
            statistics are reported for the TOTAL population, both shown and 
            unshown. This includes the histograms/KDE plots.
        fps : int, default=10
            Frames per second, only used if saving a movie to file. Make
            sure this is at least as big as 1/dt, where dt is the time interval
            between frames!
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
            n0 = 0
        else:
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
                if clip is not None:
                    norm = colors.Normalize(-abs(clip),abs(clip),clip=True)
                else:
                    norm = None
                fld = ax.pcolormesh([self.envir.flow_points[0]], self.envir.flow_points[1], 
                           np.zeros(self.envir.flow[0].shape[1:]).T, shading='gouraud',
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
                M = round(len(self.envir.flow_points[0])/x_num)
                N = round(len(self.envir.flow_points[1])/y_num)
                # get worse case max velocity vector for scaling
                max_u = self.envir.flow[0].max(); max_v = self.envir.flow[1].max()
                max_mag = np.linalg.norm(np.array([max_u,max_v]))
                x_pts = self.envir.flow_points[0][::M]
                y_pts = self.envir.flow_points[1][::N]
                fld = ax.quiver(x_pts, y_pts, np.zeros((len(y_pts),len(x_pts))),
                                np.zeros((len(y_pts),len(x_pts))), 
                                scale=max_mag*5, alpha=0.2)

            # scatter plot
            scat = ax.scatter([], [], label=self.shared_props['name'], 
                              c=self.shared_props['color'])
            
            # set up marker headings to be added to the scatter plots
            circle = Path.circle(radius=circ_rad)
            line_codes = np.array([Path.MOVETO, Path.LINETO])
            codes = np.concatenate([circle.codes, line_codes])

            # textual info
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                fontsize=12)
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_swrm_vel = \
                self._calc_basic_stats(DIM3=False, t_indx=n0)
            axStats = plt.axes([0.77, 0.77, 0.25, 0.2], frameon=False)
            axStats.set_axis_off()
            stats_text = axStats.text(0,1,
                         '{:.1f}% remain\n'.format(perc_left)+
                         '\n  ------ Info ------\n'+
                         r'Fluid $v_{max}$'+': {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                         r'Fluid $\overline{v}$'+': {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                         r'Agent $\overline{v}$'+': {:.1g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units),
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
            #   arrays. The implemenation does not parallel 2D: it wants a color 
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
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel = \
                self._calc_basic_stats(DIM3=True, t_indx=n0)
            flow_text = ax.text2D(0.75, 0.9,
                                  r'Fluid $v_{max}$'+': {:.2g} {}/s\n'.format(
                                  max_spd, self.envir.units)+
                                  r'Fluid $v_{avg}$'+': {:.2g} {}/s\n'.format(
                                  avg_spd, self.envir.units)+
                                  r'Agent $v_{avg}$'+': {:.2g} {}/s'.format(
                                  np.linalg.norm(avg_swrm_vel), self.envir.units),
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
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_swrm_vel = \
                        self._calc_basic_stats(DIM3=False, t_indx=n)
                    stats_text.set_text('{:.1f}% remain\n'.format(perc_left)+
                         '\n  ------ Info ------\n'+
                         r'Fluid $v_{max}$'+': {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                         r'Fluid $\overline{v}$'+': {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                         r'Agent $\overline{v}$'+': {:.1g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units))
                    x_text.set_text(r'Fluid $\overline{v}_x$'+': {:.2g} \n'.format(avg_spd_x)+
                         r'Agent $\overline{v}_x$'+': {:.2g}'.format(avg_swrm_vel[0]))
                    y_text.set_text(r'Fluid $\overline{v}_y$'+': {:.2g} \n'.format(avg_spd_y)+
                         r'Agent $\overline{v}_y$'+': {:.2g}'.format(avg_swrm_vel[1]))
                    if fluid == 'vort' and self.envir.flow is not None:
                        vort = self.envir.get_2D_vorticity(t_indx=n)
                        fld.set_array(vort.T)
                        fld.changed()
                        fld.autoscale()
                    elif fluid == 'quiver' and self.envir.flow is not None:
                        if self.envir.flow_times is not None:
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
                                paths.append(Path(verts, codes))
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
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel = \
                        self._calc_basic_stats(DIM3=True, t_indx=n)
                    # print(n)
                    # print(self.pos_history[n].all() is ma.masked)
                    flow_text.set_text(r'Fluid $v_{max}$'+': {:.2g} {}/s\n'.format(
                                       max_spd, self.envir.units)+
                                       r'Fluid $v_{avg}$'+': {:.2g} {}/s\n'.format(
                                       avg_spd, self.envir.units)+
                                       r'Agent $v_{avg}$'+': {:.2g} {}/s'.format(
                                       np.linalg.norm(avg_swrm_vel), self.envir.units))
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
                    #   arrays. The implemenation does not parallel 2D: it wants a color 
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
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_swrm_vel = \
                        self._calc_basic_stats(DIM3=False, t_indx=None)
                    stats_text.set_text('{:.1f}% remain\n'.format(perc_left)+
                         '\n  ------ Info ------\n'+
                         r'Fluid $v_{max}$'+': {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                         r'Fluid $\overline{v}$'+': {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                         r'Agent $\overline{v}$'+': {:.1g} {}/s\n'.format(np.linalg.norm(avg_swrm_vel), self.envir.units))
                    x_text.set_text(r'Fluid $\overline{v}_x$'+': {:.2g} \n'.format(avg_spd_x)+
                         r'Agent $\overline{v}_x$'+': {:.2g}'.format(avg_swrm_vel[0]))
                    y_text.set_text(r'Fluid $\overline{v}_y$'+': {:.2g} \n'.format(avg_spd_y)+
                         r'Agent $\overline{v}_y$'+': {:.2g}'.format(avg_swrm_vel[1]))
                    if fluid == 'vort' and self.envir.flow is not None:
                        vort = self.envir.get_2D_vorticity()
                        fld.set_array(vort.T)
                        fld.changed()
                        fld.autoscale()
                    elif fluid == 'quiver' and self.envir.flow is not None:
                        if self.envir.flow_times is not None:
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
                                paths.append(Path(verts, codes))
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
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel = \
                        self._calc_basic_stats(DIM3=True)
                    flow_text.set_text(r'Fluid $v_{max}$'+': {:.2g} {}/s\n'.format(
                                       max_spd, self.envir.units)+
                                       r'Fluid $v_{avg}$'+': {:.2g} {}/s\n'.format(
                                       avg_spd, self.envir.units)+
                                       r'Agent $v_{avg}$'+': {:.2g} {}/s'.format(
                                       np.linalg.norm(avg_swrm_vel), self.envir.units))
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
                    #   arrays. The implemenation does not parallel 2D: it wants a color 
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

        # infer animation rate from dt between current and last position
        dt = self.envir.time - self.envir.time_history[-1]

        if frames is None:
            frames = range(len(self.pos_history)+1)
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                    interval=dt*100, repeat=False, blit=True)

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

