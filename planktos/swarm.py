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
from scipy import interpolate, stats
from scipy.spatial import distance
import pandas as pd
if sys.platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, colors

from .environment import environment
from . import dataio
from . import motion

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
        - 'random': Uniform random distribution throughout the domain
        - 'grid': Uniform grid on interior of the domain, including capability
            to leave out closed immersed structures. In this case, swarm_size 
            is ignored since it is determined by the grid dimensions.
            Requires the additional keyword parameters grid_dim and testdir.
        - 1D array: All positions set to a single point given by the x,y,[z] 
            coordinates of this array
        - 2D array: All positions as specified. Shape of array should be NxD, 
            where N is the number of agents and D is spatial dimension. In this 
            case, swarm_size is ignored.
    seed : int, optional
        Seed for random number generator
    shared_props : dictionary, optional
        dictionary of properties shared by all agents as name-value pairs. If 
        none are provided, two default properties will be created, 'mu' and 'cov', 
        corresponding to intrinsic mean drift and a covariance matrix for 
        brownian motion respectively. 'mu' will be set to an array of zeros with 
        length matching the spatial dimension, and 'cov' will be set to an 
        identity matrix of appropriate size according to the spatial dimension. 
        This allows the default agent behavior to be unbiased brownian motion.  
        Examples:  
        - diam: diameter of the particles
        - m: mass of the particles
        - Cd: drag coefficient of the particles
        - cross_sec: cross-sectional area of the particles
        - R: density ratio
    props : Pandas dataframe of individual agent properties, optional
        Pandas dataframe of individual agent properties that vary between agents. 
        This is the method by which individual variation among the agents should 
        be specified. The number of rows in the dataframe should match the 
        number of agents. If no dataframe is supplied, a default one is created 
        which contains only the agent starting positions in a column entitled
        'start_pos'. This is to aid in creating more properties later, if 
        desired, as it is only necessary to add columns to the existing dataframe. 
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
    pos_history : list of masked arrays
        all previous position arrays are stored here. to get their corresponding 
        times, check the time_history attribute of the swarm's environment.
    full_pos_history : list of masked arrays
        same as pos_history, but also includes the positions attribute as the 
        last entry in the list
    velocities : masked array, shape Nx2 (2D) or Nx3 (3D)
        velocity of all the agents in the swarm. same masking properties as 
        positions
    accelerations : masked array, shape Nx2 (2D) or Nx3 (3D)
        accelerations of all the agents in the swarm. same masking properties as 
        positions
    ib_collision : 1D array of bool with length equal to the swarm size
        For each agent, True if the agent collided with an immersed boundary in
        the most recent time it moved. False otherwise.
    props : pandas DataFrame
        Pandas dataframe of individual agent properties that vary between agents. 
        This is the method by which individual variation among the agents should 
        be specified.
    shared_props : dictionary
        dictionary of properties shared by all agents as name-value pairs
    rndState : numpy RandomState object
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

    def __init__(self, swarm_size=100, envir=None, init='random', seed=None, 
                 shared_props=None, props=None, **kwargs):

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
        self.rndState = np.random.RandomState(seed=seed)

        # initialize agent locations
        if isinstance(init,np.ndarray) and len(init.shape) == 2:
            swarm_size = init.shape[0]
        self.positions = ma.zeros((swarm_size, len(self.envir.L)))
        if isinstance(init,str):
            if init == 'random':
                print('Initializing swarm with uniform random positions...')
                for ii in range(len(self.envir.L)):
                    self.positions[:,ii] = self.rndState.uniform(0, 
                                        self.envir.L[ii], self.positions.shape[0])
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
                swarm_size = self.positions.shape[0]
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

        # Initialize position history
        self.pos_history = []

        # Apply boundary conditions in case of domain mismatch
        self.apply_boundary_conditions(ib_collisions=None)

        # Initialize IB collision detection
        self.ib_collision = np.full(swarm_size, False)

        # initialize Dataframe of non-shared properties
        if props is None:
            self.props = pd.DataFrame(
                {'start_pos': [tuple(self.positions[ii,:]) for ii in range(swarm_size)]}
            )
            # with random cov
            # self.props = pd.DataFrame(
            #     {'start_pos': [tuple(self.positions[ii,:]) for ii in range(swarm_size)],
            #     'cov': [np.eye(len(self.envir.L))*(0.5+np.random.rand()) for ii in range(swarm_size)]}
            # )
        else:
            self.props = props

        # Dictionary of shared properties
        if shared_props is None:
            self.shared_props = {}
        else:
            self.shared_props = shared_props

        # Include parameters for a uniform standard random walk by default
        if 'mu' not in self.shared_props and 'mu' not in self.props:
            self.shared_props['mu'] = np.zeros(len(self.envir.L))
        if 'cov' not in self.shared_props and 'cov' not in self.props:
            self.shared_props['cov'] = np.eye(len(self.envir.L))

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
                intersections = swarm._seg_intersect_2D(pt, endpt,
                    close_mesh[:,0,:], close_mesh[:,1,:], get_all=True)
            else:
                intersections = swarm._seg_intersect_3D_triangles(pt, endpt,
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
            os.mkdir(path)

        self.save_pos_to_csv(str(path/name), pos_fmt, sv_vel=True, sv_accel=True)

        props_file = path/(name+'_props.json')
        self.props.to_json(str(props_file))
        shared_props_file = path/(name+'_shared_props.npz')
        np.savez(str(shared_props_file), **self.shared_props)



    def save_pos_to_csv(self, filename, fmt='%.18e', sv_vel=False, sv_accel=False):
        '''Save the full position history including present time, with mask and 
        time stamps, to a csv.

        The output format for the position csv will be as follows:

        - The first row contains cycle and time information. The cycle is given, 
        and then each time stamp is repeated D times, where D is the spatial 
        dimension of the system.

        - Each subsequent row corresponds to a different agent in the swarm.

        - Reading across the columns of an agent row: first, a boolean is given
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
                   np.column_stack([mat for pos in self.full_pos_history for mat in (pos[:,0].mask, pos.data)]))),
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
                data = np.zeros((self.positions[~self.positions[:,0].mask,:].shape[0],3))
                data[:,:2] = self.positions[~self.positions[:,0].mask,:]
                dataio.write_vtk_point_data(path, name, data)
            else:
                dataio.write_vtk_point_data(path, name, self.positions[~self.positions[:,0].mask,:])
        else:
            for cyc, time in enumerate(self.envir.time_history):
                if DIM2:
                    data = np.zeros((self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:].shape[0],3))
                    data[:,:2] = self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:]
                    dataio.write_vtk_point_data(path, name, data, 
                                                 cycle=cyc, time=time)
                else:
                    dataio.write_vtk_point_data(path, name, 
                        self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:], 
                        cycle=cyc, time=time)
            cyc = len(self.envir.time_history)
            if DIM2:
                data = np.zeros((self.positions[~self.positions[:,0].mask,:].shape[0],3))
                data[:,:2] = self.positions[~self.positions[:,0].mask,:]
                dataio.write_vtk_point_data(path, name, data, cycle=cyc,
                                             time=self.envir.time)
            else:
                dataio.write_vtk_point_data(path, name, 
                    self.positions[~self.positions[:,0].mask,:],
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



    def move(self, dt=1.0, params=None, ib_collisions='inelastic', update_time=True):
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
        ib_collisions : {None, 'inelastic' (default), 'sticky'}
            Type of interaction with immersed boundaries. If None, turn off all 
            interaction with immersed boundaries. In inelastic collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        update_time : bool, default=True
            whether or not to update the environment's time by dt. Probably 
            The only reason to change this to False is if there are multiple 
            swarm objects in the same environment - then you want to update 
            each before incrementing the time in the environment.

        See Also
        --------
        get_positions : 
            method that returns (but does not assign) the new positions of the 
            swarm after the time step dt, which Planktos users override in order 
            to specify their own, custom agent behavior.
        '''

        # Put current position in the history
        self.pos_history.append(self.positions.copy())

        # Check that something is left in the domain to move, and move it.
        if not np.all(self.positions.mask):
            # Update positions, preserving mask
            self.positions[:,:] = self.get_positions(dt, params)
            # Update velocity and acceleration of swarm
            velocity = (self.positions - self.pos_history[-1])/dt
            self.accelerations[:,:] = (velocity - self.velocities)/dt
            self.velocities[:,:] = velocity
            # Apply boundary conditions.
            self.apply_boundary_conditions(ib_collisions=ib_collisions)

        # Record new time
        if update_time:
            self.envir.time_history.append(self.envir.time)
            self.envir.time += dt
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
        documentation for the functions of this module for options. To access the 
        current positions of each agent, use self.positions which is a masked, 
        NxD array of agent positions where the mask refers to whether or not the 
        agent has exited the domain. self.velocities and self.accelerations will 
        provide initial velocities and accelerations for the time step for each 
        agent respectively. The get_fluid_drift method will return the fluid 
        velocity at each agent location. The get_dudt method will return the 
        time derivative of the fluid velocity at the location of each agent. The 
        get_fluid_gradient method will return the gradient of the magnitude of 
        the fluid velocity at the location of each agent.

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
        get_fluid_gradient : 
            return the gradient of the magnitude of the fluid velocity at each 
            agent
        '''

        # default behavior for Euler_brownian_motion is dift due to mu property
        #   plus local fluid velocity and diffusion given by cov property
        #   specifying the covariance matrix.
        return motion.Euler_brownian_motion(self, dt)



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
            if not DIM3:
                return np.array([0, 0])
            else:
                return np.array([0, 0, 0])
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



    def get_fluid_gradient(self, positions=None):
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
        if self.envir.grad is not None:
            if not TIME_DEP:
                flow_grad = self.envir.grad
            elif self.envir.grad_time == self.envir.time:
                flow_grad = self.envir.grad

        # Otherwise, calculate the gradient
        if flow_grad is None:
            self.envir.calculate_mag_gradient()
            flow_grad = self.envir.grad

        # Interpolate the gradient at agent positions and return
        x_grad = interpolate.interpn(self.envir.flow_points, flow_grad[0],
                                     positions, method='linear')
        y_grad = interpolate.interpn(self.envir.flow_points, flow_grad[1],
                                     positions, method='linear')
        if len(self.envir.flow_points) == 3:
            z_grad = interpolate.interpn(self.envir.flow_points, flow_grad[2],
                                         positions, method='linear')
            return np.array([x_grad, y_grad, z_grad]).T
        else:
            return np.array([x_grad, y_grad]).T



    def apply_boundary_conditions(self, ib_collisions='inelastic'):
        '''Apply boundary conditions to self.positions.
        
        There should be no reason to call this method directly; it is 
        automatically called by self.move after updating agent positions 
        according to the algorithm found in self.get_positions.

        This method compares current agent positions (self.positions) to the
        previous agent positions (last entry in self.pos_history) in order to
        first: determine if the agent collided with any immersed structures and
        if so, to update self.positions using an inelastic collision algorithm 
        based on vector projection and second: assess whether or not any agents 
        exited the domain and if so, update their positions based on the 
        boundary conditions as specified in the enviornment class (self.envir).

        For no flux boundary conditions, inelastic projections are really simple 
        (since the domain is just a box), so we just do them directly/manually
        instead of folding them into the far more complex, recursive algorithm 
        used for internal mesh structures.
        
        Parameters
        ----------
        ib_collisions : {None, 'inelastic' (default), 'sticky'}
            Type of interaction with immersed boundaries. If None, turn off all 
            interaction with immersed boundaries. In inelastic collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        '''

        # internal mesh boundaries go first
        if self.envir.ibmesh is not None and ib_collisions is not None:
            # if there is no mask, loop over all agents, appyling internal BC
            # loop over (non-masked) agents, applying internal BC
            if np.any(self.positions.mask):
                for n, startpt, endpt in \
                    zip(np.arange(self.positions.shape[0])[~self.positions.mask[:,0]],
                        self.pos_history[-1][~self.positions.mask[:,0],:].copy(),
                        self.positions[~self.positions.mask[:,0],:].copy()
                        ):
                    new_loc = self._apply_internal_BC(startpt, endpt, 
                                self.envir.ibmesh, self.envir.max_meshpt_dist,
                                ib_collisions=ib_collisions)
                    self.positions[n] = new_loc
                    if np.any(new_loc != endpt):
                        self.ib_collision[n] = True
                    else:
                        self.ib_collision[n] = False
            else:
                for n in range(self.positions.shape[0]):
                    startpt = self.pos_history[-1][n,:].copy()
                    endpt = self.positions[n,:].copy()
                    new_loc = self._apply_internal_BC(startpt, endpt,
                                self.envir.ibmesh, self.envir.max_meshpt_dist,
                                ib_collisions=ib_collisions)
                    self.positions[n] = new_loc
                    if np.any(new_loc != endpt):
                        self.ib_collision[n] = True
                    else:
                        self.ib_collision[n] = False

        for dim, bndry in enumerate(self.envir.bndry):

            # Check for 3D
            if dim == 2 and len(self.envir.L) == 2:
                # Ignore last bndry condition; 2D environment.
                break

            ### Left boundary ###
            if bndry[0] == 'zero':
                # mask everything exiting on the left
                maskrow = self.positions[:,dim] < 0
                self.positions[maskrow, :] = ma.masked
                self.velocities[maskrow, :] = ma.masked
                self.accelerations[maskrow, :] = ma.masked
            elif bndry[0] == 'noflux':
                # pull everything exiting on the left to 0
                zerorow = self.positions[:,dim] < 0
                self.positions[zerorow, dim] = 0
                self.velocities[zerorow, dim] = 0
                self.accelerations[zerorow, dim] = 0
            else:
                raise NameError

            ### Right boundary ###
            if bndry[1] == 'zero':
                # mask everything exiting on the right
                maskrow = self.positions[:,dim] > self.envir.L[dim]
                self.positions[maskrow, :] = ma.masked
                self.velocities[maskrow, :] = ma.masked
                self.accelerations[maskrow, :] = ma.masked
            elif bndry[1] == 'noflux':
                # pull everything exiting on the left to 0
                zerorow = self.positions[:,dim] > self.envir.L[dim]
                self.positions[zerorow, dim] = self.envir.L[dim]
                self.velocities[zerorow, dim] = 0
                self.accelerations[zerorow, dim] = 0
            else:
                raise NameError


    
    @staticmethod
    def _apply_internal_BC(startpt, endpt, mesh, max_meshpt_dist, 
                           old_intersection=None, kill=False, 
                           ib_collisions='inelastic'):
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
            slid along the boundary line between two mesh elements. This 
            prevents such a thing from happening more than once, in case of 
            pathological cases.
        ib_collisions : {'inelastic' (default), 'sticky'}
            Type of interaction with immersed boundaries. In inelastic 
            collisions, conduct recursive vector projection until the length of
            the original vector is exhausted. In sticky collisions, just return 
            the point of intersection.

        Returns
        -------
        newendpt : length 2 or 3 array
            new end location for agent trajectory
        '''

        if len(startpt) == 2:
            DIM = 2
        else:
            DIM = 3

        # Get the distance for inclusion of meshpoints
        traj_dist = np.linalg.norm(endpt - startpt)
        # Must add to traj_dist to find endpoints of line segments
        search_dist = np.linalg.norm((traj_dist,max_meshpt_dist*0.5))

        # Find all mesh elements that have points within this distance
        #   NOTE: This is a major bottleneck if not done carefully.
        pt_bool = np.linalg.norm(
                mesh.reshape((mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-startpt,
                axis=1)<search_dist
        pt_bool = pt_bool.reshape((mesh.shape[0],mesh.shape[1]))
        close_mesh = mesh[np.any(pt_bool,axis=1)]

        # elem_bool = [np.any(np.linalg.norm(mesh[ii]-startpt,axis=1)<search_dist)
        #              for ii in range(mesh.shape[0])]

        # Get intersections
        if DIM == 2:
            intersection = swarm._seg_intersect_2D(startpt, endpt,
                close_mesh[:,0,:], close_mesh[:,1,:])
        else:
            intersection = swarm._seg_intersect_3D_triangles(startpt, endpt,
                close_mesh[:,0,:], close_mesh[:,1,:], close_mesh[:,2,:])

        # Return endpt we already have if None.
        if intersection is None:
            return endpt
        
        # If we do have an intersection:
        if ib_collisions == 'inelastic':
            # Project remaining piece of vector onto mesh and repeat processes 
            #   as necessary until we have a final result.
            return swarm._project_and_slide(startpt, endpt, intersection,
                                            close_mesh, max_meshpt_dist, DIM,
                                            old_intersection, kill)
        elif ib_collisions == 'sticky':
            # Return the point of intersection
            
            # small number to perturb off of the actual boundary in order to avoid
            #   roundoff errors that would allow penetration
            EPS = 1e-7

            back_vec = (startpt-endpt)/np.linalg.norm(endpt-startpt)
            return intersection[0] + back_vec*EPS



    @staticmethod
    def _project_and_slide(startpt, endpt, intersection, mesh, max_meshpt_dist,
                           DIM, old_intersection=None, kill=False):
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
            result of _seg_intersect_2D or _seg_intersect_3D_triangles. various 
            information about the intersection with the immersed mesh element
        mesh : Nx2x2 or Nx3x3 array 
            eligible (nearby) mesh elements for interaction
        max_meshpt_dist : float
            max distance between two points on a mesh element (used to determine 
            how far away from startpt to search for mesh elements). Used here 
            for passthrough to possible recursion
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
            norm_out = (proj-vec)/np.linalg.norm(proj-vec)
        if DIM == 3:
            # get a unit normal vector pointing back the way we came
            norm_out = -np.sign(np.dot(vec,intersection[2]))*intersection[2]
            # Get the component of vec that lies in the plane. We do this by
            #   subtracting off the component which is in the normal direction
            proj = vec - np.dot(vec,norm_out)*norm_out
            
        # IMPORTANT: there can be roundoff error, so proj should be considered
        #   an approximation to an in-boundary slide.
        # For this reason, pull newendpt back a bit for numerical stability
        newendpt = intersection[0] + proj + EPS*norm_out

        ################################
        ###########    2D    ###########
        ################################
        if DIM == 2:
            # Detect sliding off 1D edge
            # Equivalent to going past the endpoints
            mesh_el_len = np.linalg.norm(intersection[4] - intersection[3])
            Q0_dist = np.linalg.norm(newendpt-EPS*norm_out - intersection[3])
            Q1_dist = np.linalg.norm(newendpt-EPS*norm_out - intersection[4])
            # Since we are sliding on the mesh element, if the distance from
            #   our new location to either of the mesh endpoints is greater
            #   than the length of the mesh element, we must have gone beyond
            #   the segment.
            if Q0_dist*(1+EPS) > mesh_el_len or Q1_dist*(1+EPS) > mesh_el_len:
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
                    np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-intersection[3],
                    axis=1),0),
                    np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-intersection[4],
                    axis=1),0)
                )
                pt_bool = pt_bool.reshape((mesh.shape[0],mesh.shape[1]))
                adj_mesh = mesh[np.any(pt_bool,axis=1)]
                # Check for intersection with these segemtns, but translate 
                #   start/end points back from the segment a bit for numerical stability
                # This has already been done for newendpt
                # Also go EPS further than newendpt for stability for what follows
                if len(adj_mesh) > 0:
                    adj_intersect = swarm._seg_intersect_2D(intersection[0]+EPS*norm_out,
                        newendpt+EPS*proj, adj_mesh[:,0,:], adj_mesh[:,1,:])
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
                        return adj_intersect[0] - EPS*proj
                    # Otherwise:
                    if Q0_dist*(1+EPS) > mesh_el_len and Q0_dist*(1+EPS) > Q1_dist*(1+EPS):
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
                        return adj_intersect[0] - EPS*proj
                    else:
                        # Obtuse. Repeat project_and_slide on new segment,
                        #   but send along info about old segment so we don't
                        #   get in an infinite loop.
                        return swarm._project_and_slide(intersection[0]+EPS*norm_out, 
                                                        newendpt+EPS*proj, 
                                                        adj_intersect, mesh, 
                                                        max_meshpt_dist, DIM,
                                                        intersection)

                ######  Went past, but did not intersect adjoining element! ######

                # NOTE: There could still be an adjoining element at >180 degrees
                #   But we will project along original heading as if there isn't one
                #   and subsequently detect any intersections.
                # DIM == 2, adj_intersect is None
                
                if Q0_dist >= mesh_el_len and Q0_dist >= Q1_dist:
                    ##### went past Q1 #####
                    # put a new start point at the point crossing+EPS and bring out
                    #   EPS*norm_out
                    newstartpt = intersection[4] + EPS*proj + EPS*norm_out
                elif Q1_dist >= mesh_el_len:
                    ##### went past Q0 #####
                    newstartpt = intersection[3] + EPS*proj + EPS*norm_out
                else:
                    raise RuntimeError("Impossible case??")
                
                # continue full amount of remaining movement along original heading
                #   starting at newstartpt
                orig_unit_vec = (endpt-startpt)/np.linalg.norm(endpt-startpt)
                newendpt = newstartpt + np.linalg.norm(newendpt-newstartpt)*orig_unit_vec
                # repeat process to look for additional intersections
                #   pass along current intersection in case of obtuse concave case
                return swarm._apply_internal_BC(newstartpt, newendpt,
                                                mesh, max_meshpt_dist,
                                                intersection)

            else:
                ##########      Did not go past either Q0 or Q1      ##########
                # We simply end on the mesh element
                return newendpt
        
        ################################
        ###########    3D    ###########
        ################################
        if DIM == 3:
            # Detect sliding off 2D edge using _seg_intersect_2D
            Q0_list = np.array(intersection[3:])
            Q1_list = Q0_list[(1,2,0),:]
            # go a little further along trajectory to treat acute case
            tri_intersect = swarm._seg_intersect_2D(intersection[0] + EPS*norm_out,
                                                    newendpt + EPS*proj,
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
                    np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-intersection[3],
                    axis=1),0),
                    np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-intersection[4],
                    axis=1),0)),
                    np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-intersection[5],
                    axis=1),0)
                )
                pt_bool = pt_bool.reshape((mesh.shape[0],mesh.shape[1]))
                adj_mesh = mesh[np.any(pt_bool,axis=1)]
                # check for intersection, but translate start/end points back
                #   from the simplex a bit for numerical stability
                # this has already been done for newendpt
                # also go EPS further than newendpt for stability for what follows
                if len(adj_mesh) > 0:
                    adj_intersect = swarm._seg_intersect_3D_triangles(
                        intersection[0]+EPS*norm_out,
                        newendpt+EPS*proj, adj_mesh[:,0,:],
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
                            return adj_intersect[0] - EPS*proj

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
                        newstartpt = adj_intersect[0] - EPS*proj
                        # Get leftover portion of the travel vector
                        vec_to_project = (1-adj_intersect[1])*proj
                        # project vec onto the line
                        proj_vec = np.dot(vec_to_project,vec_intersect)*vec_intersect
                        # Get new endpoint
                        newendpt = newstartpt + proj_vec
                        # Check for more intersections
                        return swarm._apply_internal_BC(newstartpt, newendpt,
                                                        mesh, max_meshpt_dist,
                                                        adj_intersect, kill)

                    # Not an already discovered mesh element.
                    # We slide. Pass along info about the old element.
                    return swarm._project_and_slide(intersection[0]+EPS*norm_out, 
                                                    newendpt+EPS*proj+EPS*norm_out, 
                                                    adj_intersect, mesh, 
                                                    max_meshpt_dist, DIM,
                                                    intersection, kill)

                ##########      Did not intersect adjoining element!      ##########
                # NOTE: There could still be an adjoining element w/ >180 connection
                #   But we will project along original heading as if there isn't
                #   one and subsequently detect any intersections.
                # DIM == 3, adj_intersect is None
                # put the new start point on the edge of the simplex (+EPS) and
                #   continue full amount of remaining movement along original heading
                newstartpt = tri_intersect[0] + EPS*proj + EPS*norm_out
                orig_unit_vec = (endpt-startpt)/np.linalg.norm(endpt-startpt)
                # norm(newendpt - tri_intersect[0]) is the length of the overshoot.
                newendpt = newstartpt + np.linalg.norm(newendpt-tri_intersect[0])*orig_unit_vec
                # repeat process to look for additional intersections
                return swarm._apply_internal_BC(newstartpt, newendpt, 
                                                mesh, max_meshpt_dist,
                                                intersection, kill)
            else:
                # otherwise, we end on the mesh element
                return newendpt



    @staticmethod
    def _seg_intersect_2D(P0, P1, Q0_list, Q1_list, get_all=False):
        '''Find the intersection between two line segments (2D objects), P and Q, 
        returning None if there isn't one or if they are parallel.

        If Q is a 2D array, loop over the rows of Q finding all intersections
        between P and each row of Q, but only return the closest intersection
        to P0 (if there is one, otherwise None)

        This works for both 2D problems and problems in which P is a 3D line 
        segment roughly lying on a plane (e.g., in cases of projection along a 
        3D triangular mesh element). The plane is described by the first two 
        vectors Q, so in this case, Q0_list and Q1_list must have at least two 
        rows. The 3D problem is robust to cases where P is not exactly in the 
        plane because the algorithm is actually checking to see if its 
        projection onto the triangle crosses any of the lines in Q. This is 
        important to deal with roundoff error.

        This algorithm uses a parameteric equation approach for speed, based on
        http://geomalgorithms.com/a05-_intersect-1.html
        
        Parameters
        ----------
        P0 : length 2 (or 3) array
            first point in line segment P
        P1 : length 2 (or 3) array
            second point in line segment P 
        Q0_list : Nx2 (Nx3) ndarray 
            first points in a list of line segments.
        Q1_list : Nx2 (Nx3) ndarray 
            second points in a list of line segments.
        get_all : bool
            Return all intersections instead of just the first one encountered 
            as one travels from P0 to P1.

        Returns
        -------
        None if there is no intersection. Otherwise:
        x : length 2 (or 3) array 
            the coordinates of the point of first intersection
        s_I : float between 0 and 1
            the fraction of the line segment traveled from P0 to P1 before
            intersection occurred
        vec : length 2 (or 3) array
            directional unit vector along the boundary (Q) intersected
        Q0 : length 2 (or 3) array
            first endpoint of mesh segment intersected
        Q1 : length 2 (or 3) array
            second endpoint of mesh segment intersected
        '''

        u = P1 - P0
        v = Q1_list - Q0_list
        w = P0 - Q0_list

        if len(P0) == 2:
            u_perp = np.array([-u[1], u[0]])
            v_perp = np.array([-v[...,1], v[...,0]]).T
        else:
            normal = np.cross(v[0],v[1])
            normal /= np.linalg.norm(normal)
            # roundoff error in only an np.dot projection can be as high as 1e-7
            assert np.isclose(np.dot(u,normal),0,atol=1e-6), "P vector not parallel to Q plane"
            u_perp = np.cross(u,normal)
            v_perp = np.cross(v,normal)


        if len(Q0_list.shape) == 1:
            # only one point in Q list
            denom = np.dot(v_perp,u)
            if denom != 0:
                s_I = np.dot(-v_perp,w)/denom
                t_I = -np.dot(u_perp,w)/denom
                if 0<=s_I<=1 and 0<=t_I<=1:
                    return (P0 + s_I*u, s_I, v, Q0_list, Q1_list)
            return None

        denom_list = np.multiply(v_perp,u).sum(1) #vectorized dot product

        # We need to deal with parallel cases. With roundoff error, exact zeros
        #   are unlikely (but ruled out below). Another possiblity is getting
        #   inf values, but these will not record as being between 0 and 1, and
        #   will not throw errors when compared to these values. So all should
        #   be good.
        # 
        # Non-parallel cases:
        not_par = denom_list != 0

        # All non-parallel lines in the same plane intersect at some point.
        #   Find the parametric values for both vectors at which that intersect
        #   happens. Call these s_I and t_I respectively.
        s_I_list = -np.ones_like(denom_list)
        t_I_list = -np.ones_like(denom_list)
        # Now only need to calculuate s_I & t_I for non parallel cases; others
        #   will report as not intersecting automatically (as -1)
        #   (einsum is faster for vectorized dot product, but need same length,
        #   non-empty vectors)
        # In 3D, we are actually projecting u onto the plane of the triangle
        #   and doing our calculation there. So no problem about numerical
        #   error and offsets putting u above the plane.
        if np.any(not_par):
            s_I_list[not_par] = np.einsum('ij,ij->i',-v_perp[not_par],w[not_par])/denom_list[not_par]
            t_I_list[not_par] = -np.multiply(u_perp,w[not_par]).sum(1)/denom_list[not_par]

        # The length of our vectors parameterizing the lines is the same as the
        #   length of the line segment. So for the line segments to have intersected,
        #   the parameter values for their intersect must both be in the unit interval.
        intersect = np.logical_and(
                        np.logical_and(0<=s_I_list, s_I_list<=1),
                        np.logical_and(0<=t_I_list, t_I_list<=1))

        if np.any(intersect):
            if get_all:
                x = []
                for s_I in s_I_list[intersect]:
                    x.append(P0+s_I*u)
                return zip(
                    x, s_I_list[intersect], 
                    v[intersect]/np.linalg.norm(v[intersect]),
                    Q0_list[intersect], Q1_list[intersect]
                )
            else:
                # find the closest intersection and return it
                Q0 = Q0_list[intersect]
                Q1 = Q1_list[intersect]
                v_intersected = v[intersect]
                s_I = s_I_list[intersect].min()
                s_I_idx = s_I_list[intersect].argmin()
                return (P0 + s_I*u, s_I,
                        v_intersected[s_I_idx]/np.linalg.norm(v_intersected[s_I_idx]),
                        Q0[s_I_idx], Q1[s_I_idx])
        else:
            return None



    @staticmethod
    def _seg_intersect_3D_triangles(P0, P1, Q0_list, Q1_list, Q2_list, get_all=False):
        '''Find the intersection between a line segment P0 to P1 and any of the
        triangles given by Q0, Q1, Q2 where each row across the three arrays is 
        a different triangle (three points).
        Returns None if there is no intersection.

        This algorithm uses a parameteric equation approach for speed, based on
        http://geomalgorithms.com/a05-_intersect-1.html

        Parameters
        ----------
        P0 : length 3 array
            first point in line segment P
        P1 : length 3 array
            second point in line segment P 
        Q0 : Nx3 ndarray 
            first points in a list of triangles.
        Q1 : Nx3 ndarray 
            second points in a list of triangles.
        Q2 : Nx3 ndarray 
            third points in a list of triangles.
        get_all : bool
            Return all intersections instead of just the first one encountered 
            as you travel from P0 to P1.

        Returns
        -------
        None if there is no intersection. Otherwise: 
        x : length 3 array 
            the coordinates of the first point of intersection
        s_I : float between 0 and 1
            the fraction of the line segment traveled from P0 before 
            intersection occurred (only if intersection occurred)
        normal : length 3 array
            normal unit vector to plane of intersection
        Q0 : length 3 array
            first vertex of triangle intersected
        Q1 : length 3 array
            second vertex of triangle intersected
        Q2 : length 3 array
            third vertex of triangle intersected
        '''

        # First, determine the intersection between the line and the plane

        # Get normal vectors
        Q1Q0_diff = Q1_list-Q0_list
        Q2Q0_diff = Q2_list-Q0_list
        n_list = np.cross(Q1Q0_diff, Q2Q0_diff)

        u = P1 - P0
        w = P0 - Q0_list

        # At intersection, w + su is perpendicular to n
        if len(Q0_list.shape) == 1:
            # only one triangle
            denom = np.dot(n_list,u)
            if denom != 0:
                s_I = np.dot(-n_list,w)/denom
                if 0<=s_I<=1:
                    # line segment crosses full plane
                    cross_pt = P0 + s_I*u
                    # calculate barycentric coordinates
                    normal = n_list/np.linalg.norm(n_list)
                    A_dbl = np.dot(n_list, normal)
                    Q0Pt = cross_pt-Q0_list
                    A_u_dbl = np.dot(np.cross(Q0Pt,Q2Q0_diff),normal)
                    A_v_dbl = np.dot(np.cross(Q1Q0_diff,Q0Pt),normal)
                    coords = np.array([A_u_dbl/A_dbl, A_v_dbl/A_dbl, 0])
                    coords[2] = 1 - coords[0] - coords[1]
                    # check if point is in triangle
                    if np.all(coords>=0):
                        return (cross_pt, s_I, normal, Q0_list, Q1_list, Q2_list)
            return None

        denom_list = np.multiply(n_list,u).sum(1) #vectorized dot product

        # record non-parallel cases
        not_par = denom_list != 0

        # default is not intersecting
        s_I_list = -np.ones_like(denom_list)
        
        # get intersection parameters
        #   (einsum is faster for vectorized dot product, but need same length vectors)
        if np.any(not_par):
            s_I_list[not_par] = np.einsum('ij,ij->i',-n_list[not_par],w[not_par])/denom_list[not_par]
        # test for intersection of line segment with full plane
        plane_int = np.logical_and(0<=s_I_list, s_I_list<=1)

        # calculate barycentric coordinates for each plane intersection
        closest_int = (None, -1, None)
        intersections = []
        for n, s_I in zip(np.arange(len(plane_int))[plane_int], s_I_list[plane_int]):
            # if get_all is False, we only care about the closest triangle intersection!
            # see if we need to worry about this one
            if closest_int[1] == -1 or closest_int[1] > s_I or get_all:
                cross_pt = P0 + s_I*u
                normal = n_list[n]/np.linalg.norm(n_list[n])
                A_dbl = np.dot(n_list[n], normal)
                Q0Pt = cross_pt-Q0_list[n]
                A_u_dbl = np.dot(np.cross(Q0Pt,Q2Q0_diff[n]),normal)
                A_v_dbl = np.dot(np.cross(Q1Q0_diff[n],Q0Pt),normal)
                coords = np.array([A_u_dbl/A_dbl, A_v_dbl/A_dbl, 0])
                coords[2] = 1 - coords[0] - coords[1]
                # check if point is in triangle
                if np.all(coords>=0):
                    if get_all:
                        intersections.append((cross_pt, s_I, normal, Q0_list[n], Q1_list[n], Q2_list[n]))
                    else:
                        closest_int = (cross_pt, s_I, normal, Q0_list[n], Q1_list[n], Q2_list[n])
        if not get_all:
            if closest_int[0] is None:
                return None
            else:
                return closest_int
        else:
            if len(intersections) == 0:
                return None
            else:
                return intersections



    @staticmethod
    def _dist_point_to_plane(P0, normal, Q0):
        '''Return the distance from the point P0 to the plane given by a
        normal vector and a point on the plane, Q0. For debugging.'''

        d = np.dot(normal, Q0)
        return np.abs(np.dot(normal,P0)-d)/np.linalg.norm(normal)



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
            vel_data = self.velocities[~self.velocities.mask.any(axis=1)]
            avg_swrm_vel = vel_data.mean(axis=0)
        elif t_indx == 0:
            avg_swrm_vel = np.zeros(len(self.envir.L))
        else:
            vel_data = (self.pos_history[t_indx] - self.pos_history[t_indx-1])/(
                        self.envir.time_history[t_indx]-self.envir.time_history[t_indx-1])
            avg_swrm_vel = vel_data.mean(axis=0)

        if not DIM3:
            # 2D flow
            # get current fluid flow info
            if len(self.envir.flow[0].shape) == 2:
                # temporally constant flow
                flow = self.envir.flow
            else:
                # temporally changing flow
                flow = self.envir.interpolate_temporal_flow(t_indx=t_indx)
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



    def plot(self, t=None, blocking=True, dist='density', fluid=None, clip=None, figsize=None):
        '''Plot the position of the swarm at time t, or at the current time
        if no time is supplied. The actual time plotted will depend on the
        history of movement steps; the closest entry in
        environment.time_history will be shown without interpolation.
        
        Parameters
        ----------
        t : float, optional
            time to plot. if None (default), the current time.
        blocking : bool, default True
            whether the plot should block execution or not
        dist : {'density' (default), 'cov', float, 'hist'}
            whether to plot Gaussian kernel density estimation or histogram.
            Options are:
            - 'density': plot Gaussian KDE using Scotts Factor from scipy.stats.gaussian_kde
            - 'cov': use the variance in each direction from self.shared_props['cov']
                to plot Gaussian KDE
            - float: plot Gaussian KDE using the given bandwidth factor to 
                multiply the KDE variance by
            - 'hist': plot histogram
        fluid : {'vort', 'quiver'}, optional
            Plot info on the fluid in the background. 2D only! If None, don't
            plot anything related to the fluid.
            Options are:
            - 'vort': plot vorticity in the background
            - 'quiver': quiver plot of fluid velocity in the background
        clip : float, optional
            if plotting vorticity, specifies the clip value for pseudocolor.
            this value is used for both negative and positive vorticity.
        figsize : tuple of length 2, optional
            figure size in inches, (width, height). default is a heurstic that 
            works... most of the time?
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
            ax, axHistx, axHisty = self.envir._plot_setup(fig)
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
            if fluid == 'vort':
                vort = self.envir.get_2D_vorticity(t_indx=loc)
                if clip is not None:
                    norm = colors.Normalize(-abs(clip),abs(clip),clip=True)
                else:
                    norm = None
                ax.pcolormesh(self.envir.flow_points[0], self.envir.flow_points[1], 
                              vort.T, shading='gouraud', cmap='RdBu',
                              norm=norm, alpha=0.9, antialiased=True)
            elif fluid == 'quiver':
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
                if len(self.envir.flow[0].shape) > 2:
                    flow = self.envir.interpolate_temporal_flow(t_indx=loc)
                else:
                    flow = self.envir.flow
                ax.quiver(self.envir.flow_points[0][::M], self.envir.flow_points[1][::N],
                          flow[0][::M,::N].T, flow[1][::M,::N].T, 
                          scale=max_mag*5, alpha=0.2)

            
            # scatter plot and time text
            ax.scatter(positions[:,0], positions[:,1], label='organism', c='darkgreen', s=3)
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
                xmesh = np.linspace(0, self.envir.L[0])
                ymesh = np.linspace(0, self.envir.L[1])
                # deal with point sources
                if np.all(self.positions[:,0] == self.positions[0,0]) and fac_x is None:
                    idx = (np.abs(xmesh - self.positions[0,0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                    axHistx.plot(xmesh, x_density)
                else:
                    x_density = stats.gaussian_kde(positions[:,0].compressed(), fac_x)
                    axHistx.plot(xmesh, x_density(xmesh))
                if np.all(self.positions[:,1] == self.positions[0,1]) and fac_y is None:
                    idx = (np.abs(ymesh - self.positions[0,1])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idx] = 1
                    axHisty.plot(y_density, ymesh)
                else:
                    y_density = stats.gaussian_kde(positions[:,1].compressed(), fac_y)
                    axHisty.plot(y_density(ymesh),ymesh)
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_xaxis().set_ticks([])

        else:
            # 3D plot
            if figsize is None:
                fig = plt.figure(figsize=(10,5))
            else:
                fig = plt.figure(figsize=figsize)
            ax, axHistx, axHisty, axHistz = self.envir._plot_setup(fig)

            # scatter plot and time text
            ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                       label='organism')
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
                xmesh = np.linspace(0, self.envir.L[0])
                ymesh = np.linspace(0, self.envir.L[1])
                zmesh = np.linspace(0, self.envir.L[2])
                # deal with point sources
                if np.all(self.positions[:,0] == self.positions[0,0]) and fac_x is None:
                    idx = (np.abs(xmesh - self.positions[0,0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                    axHistx.plot(xmesh, x_density)
                else:
                    x_density = stats.gaussian_kde(positions[:,0].compressed(), fac_x)
                    axHistx.plot(xmesh, x_density(xmesh))
                if np.all(self.positions[:,1] == self.positions[0,1]) and fac_y is None:
                    idx = (np.abs(ymesh - self.positions[0,1])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idx] = 1
                    axHisty.plot(ymesh, y_density)
                else:
                    y_density = stats.gaussian_kde(positions[:,1].compressed(), fac_y)
                    axHisty.plot(ymesh, y_density(ymesh))
                if np.all(self.positions[:,2] == self.positions[0,2]) and fac_z is None:
                    idx = (np.abs(zmesh - self.positions[0,2])).argmin()
                    z_density = np.zeros_like(zmesh); z_density[idx] = 1
                    axHistz.plot(zmesh, z_density)
                else:
                    z_density = stats.gaussian_kde(positions[:,2].compressed(), fac_z)        
                    axHistz.plot(zmesh, z_density(zmesh))
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_yaxis().set_ticks([])
                axHistz.get_yaxis().set_ticks([])

        # show the plot
        plt.show(block=blocking)



    def plot_all(self, movie_filename=None, frames=None, downsamp=None, fps=10, 
                 dist='density', fluid=None, clip=None, figsize=None):
        ''' Plot the history of the swarm's movement, incl. current time in 
        successively updating plots or saved as a movie file. A movie file is
        created if movie_filename is specified.
        
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
            - 'density': plot Gaussian KDE using Scotts Factor from scipy.stats.gaussian_kde
            - 'cov': use the variance in each direction from self.shared_props['cov']
                to plot Gaussian KDE
            - float: plot Gaussian KDE using the given bandwidth factor to 
                multiply the KDE variance by
            - 'hist': plot histogram
        fluid : {'vort', 'quiver'}, optional
            Plot info on the fluid in the background. 2D only! If None, don't
            plot anything related to the fluid.
            Options are:
            - 'vort': plot vorticity in the background
            - 'quiver': quiver plot of fluid velocity in the background
        clip : float, optional
            if plotting vorticity, specifies the clip value for pseudocolor.
            this value is used for both negative and positive vorticity.
        figsize : tuple of length 2, optional
            figure size in inches, (width, height). default is a heurstic that 
            works... most of the time? 
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
            ax, axHistx, axHisty = self.envir._plot_setup(fig)
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
            if fluid == 'vort':
                if clip is not None:
                    norm = colors.Normalize(-abs(clip),abs(clip),clip=True)
                else:
                    norm = None
                fld = ax.pcolormesh([self.envir.flow_points[0]], self.envir.flow_points[1], 
                           np.zeros(self.envir.flow[0].shape[1:]).T, shading='gouraud',
                           cmap='RdBu', norm=norm, alpha=0.9)
            elif fluid == 'quiver':
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
            scat = ax.scatter([], [], label='organism', c='darkgreen', s=3)

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
                xmesh = np.linspace(0, self.envir.L[0])
                ymesh = np.linspace(0, self.envir.L[1])
                # check and deal with point solution
                if np.all(self.pos_history[n0][:,0] == self.pos_history[n0][0,0]) and fac_x is None:
                    idx = (np.abs(xmesh - self.pos_history[n0][0,0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                    xdens_plt, = axHistx.plot(xmesh, x_density)
                else:
                    x_density = stats.gaussian_kde(self.pos_history[n0][:,0].compressed(), fac_x)
                    xdens_plt, = axHistx.plot(xmesh, x_density(xmesh))
                if np.all(self.pos_history[n0][:,1] == self.pos_history[n0][0,1]) and fac_y is None:
                    idx = (np.abs(ymesh - self.pos_history[n0][0,1])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idx] = 1
                    ydens_plt, = axHisty.plot(y_density, ymesh)
                else:
                    y_density = stats.gaussian_kde(self.pos_history[n0][:,1].compressed(), fac_y)
                    ydens_plt, = axHisty.plot(y_density(ymesh), ymesh)
                axHistx.set_ylim(top=np.max(xdens_plt.get_ydata()))
                axHisty.set_xlim(right=np.max(ydens_plt.get_xdata()))
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_xaxis().set_ticks([])
            
        else:
            ### 3D setup ###
            if figsize is None:
                fig = plt.figure(figsize=(10,5))
            else:
                fig = plt.figure(figsize=figsize)
            ax, axHistx, axHisty, axHistz = self.envir._plot_setup(fig)

            if downsamp is None:
                scat = ax.scatter(self.pos_history[n0][:,0], self.pos_history[n0][:,1],
                                self.pos_history[n0][:,2], label='organism',
                                animated=True)
            else:
                scat = ax.scatter(self.pos_history[n0][downsamp,0],
                                self.pos_history[n0][downsamp,1],
                                self.pos_history[n0][downsamp,2],
                                label='organism', animated=True)

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
                xmesh = np.linspace(0, self.envir.L[0])
                ymesh = np.linspace(0, self.envir.L[1])
                zmesh = np.linspace(0, self.envir.L[2])
                # check and deal with point solution
                if np.all(self.pos_history[n0][:,0] == self.pos_history[n0][0,0]) and fac_x is None:
                    idx = (np.abs(xmesh - self.pos_history[n0][0,0])).argmin()
                    x_density = np.zeros_like(xmesh); x_density[idx] = 1
                    xdens_plt, = axHistx.plot(xmesh, x_density)
                else:
                    x_density = stats.gaussian_kde(self.pos_history[n0][:,0].compressed(), fac_x)
                    xdens_plt, = axHistx.plot(xmesh, x_density(xmesh))
                if np.all(self.pos_history[n0][:,1] == self.pos_history[n0][0,1]) and fac_y is None:
                    idx = (np.abs(ymesh - self.pos_history[n0][0,1])).argmin()
                    y_density = np.zeros_like(ymesh); y_density[idx] = 1
                    ydens_plt, = axHisty.plot(ymesh, y_density)
                else:
                    y_density = stats.gaussian_kde(self.pos_history[n0][:,1].compressed(), fac_y)
                    ydens_plt, = axHisty.plot(ymesh, y_density(ymesh))
                if np.all(self.pos_history[n0][:,2] == self.pos_history[n0][0,2]) and fac_z is None:
                    idx = (np.abs(zmesh - self.pos_history[n0][0,2])).argmin()
                    z_density = np.zeros_like(zmesh); z_density[idx] = 1
                    zdens_plt, = axHistz.plot(zmesh, z_density)
                else:
                    z_density = stats.gaussian_kde(self.pos_history[n0][:,2].compressed(), fac_z)
                    zdens_plt, = axHistz.plot(zmesh, z_density(zmesh))
                axHistx.set_ylim(top=np.max(xdens_plt.get_ydata()))
                axHisty.set_ylim(top=np.max(ydens_plt.get_ydata()))
                axHistz.set_ylim(top=np.max(zdens_plt.get_ydata()))
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_yaxis().set_ticks([])
                axHistz.get_yaxis().set_ticks([])

        # animation function. Called sequentially
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
                    if fluid == 'vort':
                        vort = self.envir.get_2D_vorticity(t_indx=n)
                        fld.set_array(vort.T)
                        fld.changed()
                        fld.autoscale()
                    elif fluid == 'quiver':
                        if self.envir.flow_times is not None:
                            flow = self.envir.interpolate_temporal_flow(t_indx=n)
                            fld.set_UVC(flow[0][::M,::N].T, flow[1][::M,::N].T)
                        else:
                            fld.set_UVC(self.envir.flow[0][::M,::N].T, self.envir.flow[1][::M,::N].T)
                    if downsamp is None:
                        scat.set_offsets(self.pos_history[n])
                    else:
                        scat.set_offsets(self.pos_history[n][downsamp,:])
                    if dist == 'hist':
                        n_x, _ = np.histogram(self.pos_history[n][:,0].compressed(), bins_x)
                        n_y, _ = np.histogram(self.pos_history[n][:,1].compressed(), bins_y)
                        for rect, h in zip(patches_x, n_x):
                            rect.set_height(h)
                        for rect, h in zip(patches_y, n_y):
                            rect.set_width(h)
                        if fluid == 'vort':
                            return [fld, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                        else:
                            return [scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                    else:
                        if np.all(self.pos_history[n][:,0] == self.pos_history[n][0,0]) and fac_x is None:
                            idx = (np.abs(xmesh - self.pos_history[n][0,0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                            xdens_plt.set_ydata(x_density)
                        else:
                            x_density = stats.gaussian_kde(self.pos_history[n][:,0].compressed(), fac_x)
                            xdens_plt.set_ydata(x_density(xmesh))
                        if np.all(self.pos_history[n][:,1] == self.pos_history[n][0,1]) and fac_y is None:
                            idx = (np.abs(ymesh - self.pos_history[n][0,1])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idx] = 1
                            ydens_plt.set_xdata(y_density)
                        else:
                            y_density = stats.gaussian_kde(self.pos_history[n][:,1].compressed(), fac_y)
                            ydens_plt.set_xdata(y_density(ymesh))
                        axHistx.set_ylim(top=np.max(xdens_plt.get_ydata()))
                        axHisty.set_xlim(right=np.max(ydens_plt.get_xdata()))
                        if fluid == 'vort':
                            return [fld, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                        else:
                            return [scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
                    
                else:
                    # 3D
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z, avg_swrm_vel = \
                        self._calc_basic_stats(DIM3=True, t_indx=n)
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
                    if downsamp is None:
                        scat._offsets3d = (np.ma.ravel(self.pos_history[n][:,0].compressed()),
                                        np.ma.ravel(self.pos_history[n][:,1].compressed()),
                                        np.ma.ravel(self.pos_history[n][:,2].compressed()))
                    else:
                        scat._offsets3d = (np.ma.ravel(self.pos_history[n][downsamp,0].compressed()),
                                        np.ma.ravel(self.pos_history[n][downsamp,1].compressed()),
                                        np.ma.ravel(self.pos_history[n][downsamp,2].compressed()))
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
                        # check and deal with point solution
                        if np.all(self.pos_history[n][:,0] == self.pos_history[n][0,0]) and fac_x is None:
                            idx = (np.abs(xmesh - self.pos_history[n][0,0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                            xdens_plt.set_ydata(x_density)
                        else:
                            x_density = stats.gaussian_kde(self.pos_history[n][:,0].compressed(), fac_x)
                            xdens_plt.set_ydata(x_density(xmesh))
                        if np.all(self.pos_history[n][:,1] == self.pos_history[n][0,1]) and fac_y is None:
                            idx = (np.abs(ymesh - self.pos_history[n][0,1])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idx] = 1
                            ydens_plt.set_ydata(y_density)
                        else:
                            y_density = stats.gaussian_kde(self.pos_history[n][:,1].compressed(), fac_y)
                            ydens_plt.set_ydata(y_density(ymesh))
                        if np.all(self.pos_history[n][:,2] == self.pos_history[n][0,2]) and fac_z is None:
                            idx = (np.abs(zmesh - self.pos_history[n][0,2])).argmin()
                            z_density = np.zeros_like(zmesh); z_density[idx] = 1
                            zdens_plt.set_ydata(z_density)
                        else:
                            z_density = stats.gaussian_kde(self.pos_history[n][:,2].compressed(), fac_z)
                            zdens_plt.set_ydata(z_density(zmesh))
                        axHistx.set_ylim(top=np.max(xdens_plt.get_ydata()))
                        axHisty.set_ylim(top=np.max(ydens_plt.get_ydata()))
                        axHistz.set_ylim(top=np.max(zdens_plt.get_ydata()))
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
                    if fluid == 'vort':
                        vort = self.envir.get_2D_vorticity()
                        fld.set_array(vort.T)
                        fld.changed()
                        fld.autoscale()
                    elif fluid == 'quiver':
                        if self.envir.flow_times is not None:
                            flow = self.envir.interpolate_temporal_flow()
                            fld.set_UVC(flow[0][::M,::N].T, flow[1][::M,::N].T)
                        else:
                            fld.set_UVC(self.envir.flow[0][::M,::N].T, self.envir.flow[1][::M,::N].T)
                    if downsamp is None:
                        scat.set_offsets(self.positions)
                    else:
                        scat.set_offsets(self.positions[downsamp,:])
                    if dist == 'hist':
                        n_x, _ = np.histogram(self.positions[:,0].compressed(), bins_x)
                        n_y, _ = np.histogram(self.positions[:,1].compressed(), bins_y)
                        for rect, h in zip(patches_x, n_x):
                            rect.set_height(h)
                        for rect, h in zip(patches_y, n_y):
                            rect.set_width(h)
                        if fluid == 'vort':
                            return [fld, scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                        else:
                            return [scat, time_text, stats_text, x_text, y_text] + list(patches_x) + list(patches_y)
                    else:
                        if np.all(self.positions[:,0] == self.positions[0,0]) and fac_x is None:
                            idx = (np.abs(xmesh - self.positions[0,0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                            xdens_plt.set_ydata(x_density)
                        else:
                            x_density = stats.gaussian_kde(self.positions[:,0].compressed(), fac_x)
                            xdens_plt.set_ydata(x_density(xmesh))
                        if np.all(self.positions[:,1] == self.positions[0,1]) and fac_y is None:
                            idx = (np.abs(ymesh - self.positions[0,1])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idx] = 1
                            ydens_plt.set_xdata(y_density)
                        else:
                            y_density = stats.gaussian_kde(self.positions[:,1].compressed(), fac_y)
                            ydens_plt.set_xdata(y_density(ymesh))
                        if fluid == 'vort':
                            return [fld, scat, time_text, stats_text, x_text, y_text, xdens_plt, ydens_plt]
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
                    if downsamp is None:
                        scat._offsets3d = (np.ma.ravel(self.positions[:,0].compressed()),
                                        np.ma.ravel(self.positions[:,1].compressed()),
                                        np.ma.ravel(self.positions[:,2].compressed()))
                    else:
                        scat._offsets3d = (np.ma.ravel(self.positions[downsamp,0].compressed()),
                                        np.ma.ravel(self.positions[downsamp,1].compressed()),
                                        np.ma.ravel(self.positions[downsamp,2].compressed()))
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
                        if np.all(self.positions[:,0] == self.positions[0,0]) and fac_x is None:
                            idx = (np.abs(xmesh - self.positions[0,0])).argmin()
                            x_density = np.zeros_like(xmesh); x_density[idx] = 1
                            xdens_plt.set_ydata(x_density)
                        else:
                            x_density = stats.gaussian_kde(self.positions[:,0].compressed(), fac_x)
                            xdens_plt.set_ydata(x_density(xmesh))
                        if np.all(self.positions[:,1] == self.positions[0,1]) and fac_y is None:
                            idx = (np.abs(ymesh - self.positions[0,1])).argmin()
                            y_density = np.zeros_like(ymesh); y_density[idx] = 1
                            ydens_plt.set_ydata(y_density)
                        else:
                            y_density = stats.gaussian_kde(self.positions[:,1].compressed(), fac_y)
                            ydens_plt.set_ydata(y_density(ymesh))
                        if np.all(self.positions[:,2] == self.positions[0,2]) and fac_z is None:
                            idx = (np.abs(zmesh - self.positions[0,2])).argmin()
                            z_density = np.zeros_like(zmesh); z_density[idx] = 1
                            zdens_plt.set_ydata(z_density)
                        else:
                            z_density = stats.gaussian_kde(self.positions[:,2].compressed(), fac_z)
                            zdens_plt.set_ydata(z_density(zmesh))
                        fig.canvas.draw()
                        return [scat, time_text, flow_text, perc_text, x_text, 
                                y_text, z_text, xdens_plt, ydens_plt, zdens_plt]

        # infer animation rate from dt between current and last position
        dt = self.envir.time - self.envir.time_history[-1]

        if frames is None:
            frames = range(len(self.pos_history)+1)
        anim = animation.FuncAnimation(fig, animate, frames=frames,
                                    interval=dt*100, repeat=False, blit=True,
                                    save_count=len(frames))

        if movie_filename is not None:
            try:
                writer = animation.FFMpegWriter(fps=fps, 
                    metadata=dict(artist='Christopher Strickland'))#, bitrate=1800)
                anim.save(movie_filename, writer=writer,dpi=150)
                plt.close()
                print('Video saved to {}.'.format(movie_filename))
            except:
                print('Failed to save animation.')
                print('Check that you have ffmpeg or mencoder installed.')
                print('If you are using Anaconda ffmpeg, check that it comes from')
                print('  the conda-forge channel, as the default channel does not')
                print('  include the H.264 encoder and is thus somewhat useless.')
                raise
        else:
            plt.show()

