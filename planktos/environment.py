'''
Environment class of Planktos.

Created on Tues Jan 24 2017

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import sys, warnings
from pathlib import Path
import copy
from math import exp, log, sqrt
import numpy as np
import numpy.ma as ma
from itertools import combinations
from scipy import interpolate, stats
from scipy.spatial import distance, ConvexHull
if sys.platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.ticker import NullFormatter, MaxNLocator
from mpl_toolkits import mplot3d
from matplotlib.collections import LineCollection

import planktos
from . import dataio
from . import motion

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class environment:
    '''
    Rectangular environment containing fluid info, immersed meshs, and swarms.

    The environment class does much of the heavy lifting of Planktos. It loads 
    and contains information about the fluid velocity field, the dimensions of 
    the physical environment being simulated, boundary conditions for the agents, 
    the agent swarms that are in the environment, any immersed meshes, and all 
    simulation time information. Additionally, it provides functions for 
    manipulating the fluid velocity field in certain ways (e.g. extending,  
    tiling, and interpolating), calculating vorticity and FTLE, viewing info 
    about the fluid itself, and calling the move function on all swarms 
    contained in the environment. It is essential to any Planktos simulation 
    and typically the first Planktos object you create.

    Parameters
    ----------
    Lx, Ly : float, default=10
        Length of domain in x and y direction, meters
    Lz : float, optional
        Length of domain in z direction
    x_bndry : {'zero', 'noflux'} as str or [str, str], default='zero'
        agent boundary condition in the x-axis (if the same on both sides), or 
        [left bndry condition, right bndry condition]. Choices are 'zero' and 
        'noflux'. Agents leaving a zero boundary condition will be marked as 
        masked and cease to be updated or plotted afterward. In the noflux case, 
        agents will undergo an inelastic collision with the boundary. Movement 
        that would have occurred through the boundary will be projected onto 
        the boundary instead.
    y_bndry : str or [str, str], default='zero'
        agent boundary condition in the y-axis (if the same on both sides), or 
        [left bndry condition, right bndry condition].
    z_bndry : str or [str, str], default='noflux'
        agent boundary condition in the z-axis (if the same top and bottom), or 
        [low bndry condition, high bndry condition].
    flow : list of ndarrays, optional
        This only needs to be specified if you already have a fluid velocity field 
        loaded as a list of numpy arrays and wish to add it to the environment 
        directly. Otherwise, the fluid velocity field will be assumed to be zero 
        everywhere until something is loaded or created via a method of this class. 
        If you specify a fluid velocity field here, it should be of the following 
        format:  
        [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k]),
        z-vel field ndarray if 3D].
        Note! i is x index, j is y index, with the value of x and y increasing
        as the index increases. It is assumed that the flow mesh is equally 
        spaced and includes values on the domain boundary. A keyword argument 
        'flow_points' must also be specified as a tuple (len==dimension) of 1D 
        arrays specifying the mesh points along each direction. If the velocity 
        field is time varying, the argument 'flow_points' must also be 
        given, which should in this case be an interable of times at which the 
        fluid velocity is specified (as indexed by t in the first dimension of 
        each of the fluid ndarrays).
    flow_times : [float, float] or increasing iterable of floats, optional
        [tstart, tend] or iterable of times at which the fluid velocity 
        is specified or scalar dt; required if flow is time-dependent.
    rho : float, optional
        fluid density of environment, kg/m**3 (optional, m here meaning length units).
        Auto-calculated if mu and nu are provided.
    mu : float, optional 
        dynamic viscosity, kg/(m*s), Pa*s, N*s/m**2 (optional, m here meaning length units).
        Auto-calculated if rho and nu are provided.
    nu : float, optional 
        kinematic viscosity, m**2/s (optional, m here meaning length units).
        Auto-calculated if rho and mu are provided.
    char_L : float, optional 
        characteristic length scale. Used for calculating Reynolds
        number, especially in the case of immersed structures (ibmesh)
        and/or when simulating inertial particles
    U : float, optional
        characteristic fluid speed. Used for some calculations.
    init_swarms : swarm object or list of swarm objects, optional
        initial swarms in this environment. Can be added later.
    units : string, default='m'
        length units to use, default is meters. Note that you will
        manually need to change self.g (accel due to gravity) if using
        something else. Also, none of the methods of this class use this 
        attribute in any way, so it's probably best to work in meters.

    Attributes
    ----------
    L : list of floats
        length of the domain [x, y, [z]] in the stated units
    units : str
        A place to remind yourself of the spatial units you are working in. Note 
        that none of the methods of this class use this attribute in any way, 
        so it's probably best to work in meters.
    bndry :  list of lists, each with two of {'zero', 'noflux'}
        Boundary conditions in each direction [x, y, [z]] for agents
    swarms : list of swarm objects
        The swarms that belong to this environment
    time : float
        current environment time
    time_history : list of floats
        list of past time states
    flow : list of ndarrays
        [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k]),
            z-vel field ndarray (if 3D)]. i is x index, j is y index, with the 
            value of x and y increasing as the index increases.
    flow_times : ndarray of floats or None
        if specified, the time stamp for each index t in the flow arrays (time 
        varying fluid velocity fields only)
    flow_points : tuple (len==dimension) of 1D ndarrays
        points defining the spatial grid for the fluid velocity data
    fluid_domain_LLC : tuple (len==dimension) of floats
        original lower-left corner of domain (if from data)
    tiling : tuple (x,y) of floats 
        how much tiling was done in the x and y direction
    orig_L : tuple (Lx,Ly) of floats
        length of the domain in x and y direction (Lx,Ly) before tiling occured
    h_p : float
        height of porous region in analytic flows, e.g. Brinkman
    g : float
        accel due to gravity (length units/s**2). Only active for models of 
        motion that include gravity (this is not the default)
    rho : float
        fluid density, kg/m**3
    mu : float
        dynamic viscosity, kg/(m*s)
    nu : float
        kinematic viscosity, m**2/s
    char_L : float
        characteristic length, m
    U : float
        characteristic fluid speed, m/s
    ibmesh : ndarray
        This is either an Nx2x2 array (2D, line mesh) or an Nx3x3 array (3D, 
        triangular mesh) specifying internal mesh structures that agents will 
        treat as solid barriers. Each value of the first index into the array 
        specifies a mesh element, and each 1x2 or 1x3 contained in that index 
        is a point in space (so, two for a line, three for a triangle).
    max_meshpt_dist : float
        maximum length of a mesh segment in ibmesh, typically determined 
        automatically. This is used in search algorithms to winnow down the 
        number of mesh elements to search for intersections of movement.
    struct_plots : list of function handles
        List of functions that plot additional environment structures which the 
        agents are unaware of. E.g., for visual purposes only.
    struct_plot_args : List of tuples
        List of argument tuples to be passed to struct_plots functions, after 
        ax (the plot axis object) is passed
    FTLE_largest : ndarray
        FTLE field calculated using the largest eigenvalue
    FTLE_smallest : ndarray
        FTLE field calculated using the smallest eigenvalue. take the negative 
        of this to get backward-time information
    FTLE_loc : ndarray
        spatial points on which the FTLE mesh was calculated
    FTLE_t0 : float
        start-time for the FTLE calculation
    FTLE_T : float
        integration time for FTLE
    FTLE_grid_dim : tuple of int
        grid dimension for the FTLE calculation (x,y,[z])
    grad : ndarray
        acts as a cache for the gradient of magnitude of the fluid velocity
    grad_time : float
        simulation time at which gradient above was calculated
    t_interp : list of PPoly objects
        Used for temporal CubicSpline interpolation. Set by interpolater method.
    dt_interp : list of PPoly objects
        Used for temporal derivative interpolation. Set by interpolater method.

    Examples
    --------
    Create a 3D environment that is 10x10x10 meters with fluid density and 
    dynamic viscosity recorded. The fluid velocity is zero everywhere, but can
    be set to something different later.

    >>> envir = planktos.environment(Lz=10, rho=1000, mu=1000)

    Create a 2D 5x3 meter environment with zero fluid velocity.

    >>> envir = planktos.environment(Lx=5, Ly=3)
    '''

    def __init__(self, Lx=10, Ly=10, Lz=None,
                 x_bndry='zero', y_bndry='zero', z_bndry='noflux', flow=None,
                 flow_times=None, rho=None, mu=None, nu=None, char_L=None, 
                 U=None, init_swarms=None, units='m'):

        # Save domain size, units
        if Lz is None:
            self.L = [Lx, Ly]
        else:
            self.L = [Lx, Ly, Lz]
        self.units = units

        # Parse boundary conditions
        self.bndry = []
        self.set_boundary_conditions(x_bndry, y_bndry, z_bndry)

        ##### Fluid velocity field variables #####
        self.flow_times = None
        self.flow_points = None # tuple (len==dim) of 1D arrays specifying the mesh
        self.fluid_domain_LLC = None # original lower-left corner, if fluid comes from data
        self.tiling = None # (x,y) tiling amount
        self.orig_L = None # (Lx,Ly) before tiling/extending
        self.grad = None
        self.grad_time = None
        self.flow = flow
        self.t_interp = None # list of PPoly objs for temporal CubicSpline interpolation
        self.dt_interp = None # list of PPoly objs for temporal derivative interpolation

        if flow is not None:
            try:
                assert isinstance(flow, list)
                assert len(flow) > 1, 'flow must be specified for each dimension'
                for ii in range(len(flow)):
                    assert isinstance(flow[ii], np.ndarray)
            except AssertionError:
                tb = sys.exc_info()[2]
                raise AttributeError(
                    'flow must be specified as a list of ndarrays.').with_traceback(tb)
            if Lz is not None:
                # 3D flow
                assert len(flow) == 3, 'Must specify flow in x, y and z direction'
                if max([len(f.shape) for f in flow]) > 3:
                    # time-dependent flow
                    assert flow[0].shape[0] == flow[1].shape[0] == flow[2].shape[0]
                    assert flow_times is not None, "Must provide flow_times"
                    self.__set_flow_variables(flow_times)
                else:
                    self.__set_flow_variables()
            else:
                # 2D flow
                if max([len(f.shape) for f in flow]) > 2:
                    # time-dependent flow
                    assert flow[0].shape[0] == flow[1].shape[0]
                    assert flow_times is not None, "Must provide flow_times"
                    self.__set_flow_variables(flow_times)
                else:
                    self.__set_flow_variables()

        ##### swarm list #####

        if init_swarms is None:
            self.swarms = []
        else:
            if isinstance(init_swarms, list):
                self.swarms = init_swarms
            else:
                self.swarms = [init_swarms]
            # reset each swarm's environment
            for sw in self.swarms:
                sw.envir = self

        ##### Fluid Variables #####

        # rho: Fluid density kg/m**3
        # mu: Dynamic viscosity kg/(m*s)
        # nu: Kinematic viscosity m**2/s
        if rho == 0 or mu == 0 or nu == 0:
            raise RuntimeError("Viscosity and density of fluid cannot be zero.")
        if rho is not None and mu is not None:
            self._rho = rho
            self._mu = mu
            self._nu = mu/rho
        elif rho is not None and nu is not None:
            self._rho = rho
            self._mu = nu*rho
            self._nu = nu
        elif mu is not None and nu is not None:
            self._rho = mu/nu
            self._mu = mu
            self._nu = nu
        else:
            self._rho = rho
            self._mu = mu
            self._nu = nu

        # characteristic length
        self.char_L = char_L
        # characteristic fluid speed
        self.U = U
        # porous region height
        self.h_p = None
        # accel due to gravity
        self.g = 9.80665 # m/s**2

        ##### Immersed Boundary Mesh #####

        # When we implement a moving mesh, use np.unique to return both
        #   unique vertex values in the ibmesh AND unique_inverse, the indices
        #   to reconstruct the mesh from the unique array. Then can update
        #   points and reconstruct.
        self.ibmesh = None # Nx2x2 or Nx3x3 (element, pt in element, (x,y,z))
        self.max_meshpt_dist = None # max length of a mesh segment

        ##### Environment Structure Plotting #####

        # NOTE: the agents do not interact with these structures; for plotting only!

        # List of functions that plot additional environment structures
        self.struct_plots = []
        # List of arguments tuples to be passed to these functions, after ax
        self.struct_plot_args = []

        ##### Initalize Time #####

        # By default, time is updated whenever an individual swarm moves (swarm.move()),
        #   or when all swarms in the environment are collectively moved.
        self.time = 0.0
        self.time_history = []

        ##### FTLE fields #####
        self.FTLE_largest = None
        self.FTLE_smallest = None # take negative to get backward-time picture
        self.FTLE_loc = None
        self.FTLE_t0 = None
        self.FTLE_T = None
        self.FTLE_grid_dim = None



    @property
    def fluid_domain_LLC(self):
        return self._fluid_domain_LLC

    @fluid_domain_LLC.setter
    def fluid_domain_LLC(self, LLC):
        if LLC is not None and self.ibmesh is not None:
            # move any ibmesh to match the LLC shift
            for ii in range(len(LLC)):
                self.ibmesh[:,:,ii] -= LLC[ii]
        self._fluid_domain_LLC = LLC



    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, m):
        self._mu = m
        if self.rho is not None and self.nu is None:
            self._nu = self.mu/self.rho
        elif self.rho is None and self.nu is not None:
            self._rho = self.mu/self.nu
        elif self.rho is not None and self.nu is not None:
            warnings.warn("Both nu and rho are already set. "+
                "Skipping auto-update, please verify all values for consistency!")

        

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, n):
        self._nu = n
        if self.rho is not None and self.mu is None:
            self._mu = self.nu/self.rho
        elif self.rho is None and self.mu is not None:
            self._rho = self.mu/self.nu
        elif self.rho is not None and self.mu is not None:
            warnings.warn("Both mu and rho are already set. "+
                "Skipping auto-update, please verify all values for consistency!")



    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, r):
        self._rho = r
        if self.mu is not None and self.nu is None:
            self._nu = self.mu/self.rho
        elif self.mu is None and self.nu is not None:
            self._mu = self.nu/self.rho
        elif self.mu is not None and self.nu is not None:
            warnings.warn("Both mu and nu are already set. "+
                "Skipping auto-update, please verify all values for consistency!")



    def set_brinkman_flow(self, alpha, h_p, U, dpdx, res=101, tspan=None):
        r'''Get a fully developed Brinkman flow with a porous region.

        This method sets the environment fluid velocity as a 1D Brinkman flow 
        based on a porous layer of hight h_p in the bottom of the domain.
        Velocity gradient is zero in the x-direction and all flow moves parallel 
        to the x-axis. Porous region is the lower part of the y-domain (2D) or
        z-domain (3D) with width h_p and an empty region above. For 3D flow, the
        velocity profile is the same on all slices y=c. The decision to set
        2D vs. 3D flow is based on the current dimension of the environment.

        After this method is successfully called, the flow property of the 
        environment class will be set to the resulting Brinkman flow, and h_p 
        will be set in the environment's properties.

        Parameters
        ----------
        alpha : float
            equal to 1/(hydraulic permeability). alpha=0 implies free flow 
            (infinitely permeable)
        h_p : float
            height of porous region
        U : float or list of floats
            velocity at top of domain (v in input3d of IB2d). If a list of 
            floats, will create a time varying fluid velocity field with 
            Brinkman flow matching each U at a series of time points. The time 
            points are determined by tspan.
        dpdx : float or list of floats
            dp/dx change in momentum constant. if a list, will correspond to 
            a time varying flow field with those values of dp/dx.
        res : int
            resolution of the flow; that is, number of points at which to 
            resolve the flow, including boundaries
        tspan : [float, float] or iterable of floats, optional
            corresponds to [tstart, tend] (start time and end time with an 
            evenly spaced time mesh) or an iterable of times at which flow is 
            specified in the case of a time-varying flow field. if not specified 
            and U/dpdx are iterable, dt=1 will be used with a start time of zero.

        Examples
        --------
        Create a 3D environment with time varying Brinkman flow

        >>> envir = planktos.environment(Lz=10, rho=1000, mu=1000)
        >>> U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))
        >>> envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306, 
            res=101, tspan=[0, 20])

        Notes
        -----
        Brinkman's equation [1]_ is written as
        
        .. math::
            \rho(\mathbf{u}_{t}(\mathbf{x},t) +\mathbf{u}(\mathbf{x},t)\cdot\nabla 
            \mathbf{u}(\mathbf{x},t)) = -\nabla p(\mathbf{x},t) + \mu \nabla^2 
            \mathbf{u}(\mathbf{x},t) - \alpha \mu \mathbf{u}(\mathbf{x},t)

        where is :math:`\alpha` is the inverse of the hydraulic permeability. 
        We take a region of height h_p where :math:`\alpha>0` with parallel shear 
        flow above it, and we assume that the flow is steady 
        (:math:`\partial u/\partial t=0`), fully developed 
        (:math:`\partial u/\partial x=0`), and zero in all cross-stream directions. 
        In this case, the equations can be reduced to an analytical solution, 
        which is what we evaluate here. See [2]_ for more information.

        References
        ----------
        .. [1] H.C. Brinkman, (1949). "A calculation of the viscous force exerted 
           by a flowing fluid on a dense swarm of particles," Applied Scientific 
           Research 1, 27(34).
        .. [2] C. Strickland, L.A. Miller, A. Santhanakrishnan, C. Hamlet, 
           N.A. Battista, V. Pasour (2017). "Three-dimensional low Reynolds 
           number flows near biological filtering and protective layers," Fluids, 
           2(62).

        See Also
        --------
        set_two_layer_channel_flow
        set_canopy_flow
        '''

        ##### Parse parameters #####
        if tspan is not None and not hasattr(U, '__iter__'):
            warnings.warn("tspan specified but U is constant: Flow will be constant in time.", UserWarning)
        assert tspan is None or (hasattr(tspan, '__iter__') and len(tspan) > 1), 'tspan not recognized.'
        if hasattr(U, '__iter__'):
            try:
                assert hasattr(dpdx, '__iter__')
                assert len(dpdx) == len(U)
            except AssertionError:
                print('dpdx must be the same length as U.')
                raise
            if tspan is not None and len(tspan) > 2:
                assert len(tspan) == len(dpdx), "tspan must be same length as U/dpdx"
        else:
            assert not hasattr(dpdx, '__iter__'), 'dpdx must be the same length as U.'
            U = [U]
            dpdx = [dpdx]

        if self.mu is None or self.rho is None:
            print('Fluid properties of environment are unspecified.')
            print('mu = {}'.format(self.mu))
            print('Rho = {}'.format(self.rho))
            raise AttributeError

        ##### Calculate constant parameters #####

        # Get y-mesh
        if len(self.L) == 2:
            y_mesh = np.linspace(0, self.L[1], res)
            b = self.L[1] - h_p
        else:
            y_mesh = np.linspace(0, self.L[2], res)
            b = self.L[2] - h_p

        ##### Calculate flow velocity #####
        flow = np.zeros((len(U), res, res)) # t, y, x
        t = 0
        for v, px in zip(U, dpdx):
            # Check for v = 0
            if v != 0:
                # Calculate C and D constants and then get A and B based on these

                D = ((exp(-alpha*h_p)/(alpha**2*self.mu)*px - exp(-2*alpha*h_p)*(
                    v/(1+alpha*b)+(2-alpha**2*b**2)*px/(2*alpha**2*self.mu*(1+alpha*b))))/
                    (1-exp(-2*alpha*h_p)*((1-alpha*b)/(1+alpha*b))))

                C = (v/(1+alpha*b) + (2-alpha**2*b**2)*px/((2*alpha**2*self.mu)*(1+alpha*b)) - 
                    D*(1-alpha*b)/(1+alpha*b))

                A = alpha*C - alpha*D
                B = C + D - px/(alpha**2*self.mu)

                for n, z in enumerate(y_mesh-h_p):
                    if z > 0:
                        #Region I
                        flow[t,n,:] = z**2*px/(2*self.mu) + A*z + B
                    else:
                        #Region 2
                        if C > 0 and D > 0:
                            flow[t,n,:] = exp(log(C)+alpha*z) + exp(log(D)-alpha*z) - px/(alpha**2*self.mu)
                        elif C <= 0 and D > 0:
                            flow[t,n,:] = exp(log(D)-alpha*z) - px/(alpha**2*self.mu)
                        elif C > 0 and D <= 0:
                            flow[t,n,:] = exp(log(C)+alpha*z) - px/(alpha**2*self.mu)
                        else:
                            flow[t,n,:] = -px/(alpha**2*self.mu)
            else:
                print('U=0: returning zero flow for these time-steps.')
            t += 1

        flow = flow.squeeze() # This is 2D Brinkman flow, either t,y,x or y,x.

        ##### Set flow in either 2D or 3D domain #####

        if len(self.L) == 2:
            # 2D
            if len(flow.shape) == 2:
                # no time; (y,x) -> (x,y) coordinates
                flow = flow.T
                self.flow = [flow, np.zeros_like(flow)] #x-direction, y-direction
                self.__set_flow_variables()
            else:
                #time-dependent; (t,y,x)-> (t,x,y) coordinates
                flow = np.transpose(flow, axes=(0, 2, 1))
                self.flow = [flow, np.zeros_like(flow)]
                if tspan is None:
                    self.__set_flow_variables(tspan=1)
                else:
                    self.__set_flow_variables(tspan=tspan)
        else:
            # 3D
            if len(flow.shape) == 2:
                # (z,x) -> (x,y,z) coordinates
                flow = np.broadcast_to(flow, (res, res, res)) #(y,z,x)
                flow = np.transpose(flow, axes=(2, 0, 1)) #(x,y,z)
                self.flow = [flow, np.zeros_like(flow), np.zeros_like(flow)]

                self.__set_flow_variables()

            else:
                # (t,z,x) -> (t,z,y,x) coordinates
                flow = np.broadcast_to(flow, (res,flow.shape[0],res,res)) #(y,t,z,x)
                flow = np.transpose(flow, axes=(1, 3, 0, 2)) #(t,x,y,z)
                self.flow = [flow, np.zeros_like(flow), np.zeros_like(flow)]

                if tspan is None:
                    self.__set_flow_variables(tspan=1)
                else:
                    self.__set_flow_variables(tspan=tspan)
        self.__reset_flow_variables()
        self.h_p = h_p



    def __set_flow_variables(self, tspan=None):
        '''Store points at which flow is specified, and time information.

        Parameters
        ----------
            tspan : float, floats [tstart, tend], or iterable, optional
                times at which flow is specified or scalar dt. Required if flow 
                is time-dependent; None will be interpreted as non time-dependent 
                flow.
        '''

        # Get points defining the spatial grid for flow data
        points = []
        if tspan is None:
            # no time-dependent flow
            for dim, mesh_size in enumerate(self.flow[0].shape[::-1]):
                points.append(np.linspace(0, self.L[dim], mesh_size))
        else:
            # time-dependent flow
            for dim, mesh_size in enumerate(self.flow[0].shape[:0:-1]):
                points.append(np.linspace(0, self.L[dim], mesh_size))
        self.flow_points = tuple(points)

        # set time
        if tspan is not None:
            if not hasattr(tspan, '__iter__'):
                # set flow_times based off zero
                self.flow_times = np.arange(0, tspan*self.flow[0].shape[0], tspan)
            elif len(tspan) == 2:
                self.flow_times = np.linspace(tspan[0], tspan[1], self.flow[0].shape[0])
            else:
                assert len(tspan) == self.flow[0].shape[0]
                self.flow_times = np.array(tspan)
        else:
            self.flow_times = None



    def set_two_layer_channel_flow(self, a, h_p, Cd, S, res=101):
        '''Apply wide-channel flow with vegetation layer according to the
        two-layer model described in Defina and Bixio (2005), 
        "Vegetated Open Channel Flow" [3]_. 
        
        The decision to set 2D vs. 3D flow is based on the current dimension of
        the environment and the fluid velocity is always time-independent.

        Parameters
        ----------
        a : float
            vegetation density, given by Az*m, where Az is the frontal area of 
            vegetation per unit depth and m the number of stems per unit area 
            (1/m), assumed constant
        h_p : float
            plant height (m)
        Cd : float
            drag coefficient, assumed uniform (unitless)
        S : float 
            bottom slope (unitless, 0-1 with 0 being no slope, resulting in no flow)
        res : int 
            number of points at which to resolve the flow, including boundaries

        See Also
        --------
        set_brinkman_flow
        set_canopy_flow

        Notes
        -----
        In addition to self.flow, this will set self.h_p = a

        References
        ----------
        .. [3] A. Defina and A.C. Bixio, (2005). "Mean flow and turbulence in 
           vegetated open channel flow," Water Resources Research, 41(7).
        '''
        # Get channel height
        H = self.L[-1]
        # Get empirical length scale, Meijer and Van Valezen (1999)
        alpha = 0.0144*sqrt(H*h_p)
        # dimensionless beta
        beta = sqrt(Cd*a*h_p**2/alpha)
        # Von Karman's constant
        chi = 0.4

        # Get y-mesh
        y_mesh = np.linspace(0, H, res)

        # Find layer division
        h_p_index = np.searchsorted(y_mesh,h_p)

        # get flow velocity profile for vegetation layer
        u = np.zeros_like(y_mesh)
        u[:h_p_index] = np.sqrt(2*self.g*S/(beta*alpha/h_p**2)*(
            (H/h_p-1)*np.sinh(beta*y_mesh[:h_p_index]/h_p)/np.cosh(beta) + 1/beta))

        # compute u_h_p
        u_h_p = np.sqrt(2*self.g*S/(beta*alpha/h_p**2)*(
                        (H/h_p-1)*np.sinh(beta)/np.cosh(beta) + 1/beta))
        # estimate du_h_p/dz
        delta_z = 0.000001
        u_h_p_lower = np.sqrt(2*self.g*S/(beta*alpha/h_p**2)*(
            (H/h_p-1)*np.sinh(beta*(h_p-delta_z)/h_p)/np.cosh(beta) + 1/beta))
        du_h_pdz = (u_h_p-u_h_p_lower)/delta_z

        # calculate h_s parameter
        h_s = (self.g*S+np.sqrt((self.g*S)**2 + (4*(chi*du_h_pdz)**2)
               *self.g*S*(H-h_p) ))/(2*chi**2*du_h_pdz**2)

        # friction velocity, Klopstra et al. (1997), Nepf and Vivoni (2000)
        d = h_p - h_s
        u_star = np.sqrt(self.g*S*(H-d))

        # calculate z_0 parameter
        z_0 = h_s*np.exp(-chi*u_h_p/u_star)

        # get flow velocity profile for surface layer
        u[h_p_index:] = u_star/chi*np.log((y_mesh[h_p_index:]-d)/z_0)

        # broadcast to flow
        if len(self.L) == 2:
            # 2D
            self.flow = [np.broadcast_to(u,(res,res)), np.zeros((res,res))]
        else:
            # 3D
            self.flow = [np.broadcast_to(u,(res,res,res)), np.zeros((res,res,res)),
                         np.zeros((res,res,res))]

        # housekeeping
        self.__set_flow_variables()
        self.__reset_flow_variables()
        self.fluid_domain_LLC = None
        self.h_p = h_p



    def set_canopy_flow(self, h, a, u_star=None, U_h=None, beta=0.3, C=0.25,
                        res=101, tspan=None):
        '''
        Apply flow within and above a uniform homogenous canopy according to 
        the model described in Finnigan and Belcher (2004), "Flow over a hill 
        covered with a plant canopy" [4]_. 
        
        The decision to set 2D vs. 3D flow is based on the current dimension of
        the environment. Default values for beta and C are based on Finnigan & 
        Belcher [4]_. Must specify two of u_star, U_h, and beta, though beta has 
        a default value of 0.3 so just giving u_star or U_h will also work. 
        If one of u_star, U_h, or beta is given as a list-like object, the flow 
        will vary in time.

        Parameters
        ----------
        h : float
            height of canopy (m)
        a : float
            leaf area per unit volume of space $m^{-1}$. Typical values are a=1.0 
            for dense spruce to a=0.1 for open woodland
        u_star : float, optional (may set U_h instead)
            canopy friction velocity. u_star = U_h*beta if not set
        U_h : float, optional (may set u_star instead) 
            wind speed at top of canopy. U_h = u_star/beta if not set
        beta : float, default=0.3
            mass flux through the canopy (u_star/U_h)
        C : float, default=0.25
            drag coefficient of indivudal canopy elements
        res : int
            number of points at which to resolve the flow, including boundaries
        tspan : [float, float] or iterable of floats, optional
            [tstart, tend] or iterable of times at which flow is specified
            if None and u_star, U_h, and/or beta are iterable, dt=1 will be used.

        See Also
        --------
        set_brinkman_flow
        set_two_layer_channel_flow

        Notes
        -----
        In addition to self.flow, this will set self.h_p = h

        References
        ----------
        .. [4] J.J. Finnigan and S.E. Belcher, (2004). "Flow over a hill covered 
            with a plant canopy," Quarterly Journal of the Royal Meteorological 
            Society, 130(596), 1-29.

        '''

        ##### Parse parameters #####
        # Make sure that at least two of the three flow parameters have been specified
        none_num = 0
        if u_star is None:
            none_num +=1
        if U_h is None:
            none_num +=1
        if beta is None:
            none_num +=1
        assert none_num < 2, "At least two of u_star, U_h, and beta must be specified."
        # If tspan is given, check that a flow parameter varies.
        if tspan is not None and not (hasattr(u_star, '__iter__') or 
            hasattr(U_h, '__iter__') or hasattr(beta, '__iter__')):
            warnings.warn("tspan specified but parameters are constant: Flow will be constant in time.", UserWarning)
        assert tspan is None or (hasattr(tspan, '__iter__') and len(tspan) > 1), 'tspan format not recognized.'
        # Sanity checks for time varying flow
        iter_length = None # variable for time length
        if hasattr(u_star, '__iter__') or hasattr(U_h, '__iter__') or hasattr(beta, '__iter__'):
            # check that less than two are scalars, otherwise relation won't hold.
            # Also get the number of time points
            scal_num = 0
            if not hasattr(u_star, '__iter__') and u_star is not None:
                scal_num +=1
            elif u_star is not None:
                iter_length = len(u_star)
            if not hasattr(U_h, '__iter__') and U_h is not None:
                scal_num +=1
            elif iter_length is not None and U_h is not None:
                assert len(U_h) == iter_length, "u_star and U_h must be the same length"
            elif U_h is not None:
                iter_length = len(U_h)
            if not hasattr(beta, '__iter__') and beta is not None:
                scal_num +=1
            elif iter_length is not None and beta is not None:
                assert len(beta) == iter_length, "non-scalar flow parameters must be the same length"
            elif beta is not None:
                iter_length = len(beta)
            assert scal_num < 2, "Only one of u_star, U_h, and beta can be scalars for time varying flow"
            if tspan is not None and len(tspan) > 2:
                assert len(tspan) == iter_length, "tspan must be same length as time-varying parameters"
        elif tspan is not None:
            iter_length = len(tspan)
        # Convert flow parameters to numpy arrays to deal with inf/nan division
        if u_star is not None:
            u_star = np.array([u_star], dtype=np.float64).squeeze()
        if U_h is not None:
            U_h = np.array([U_h], dtype=np.float64).squeeze()
        if beta is not None:
            beta = np.array([beta], dtype=np.float64).squeeze()

        # Get domain height
        d_height = self.L[-1]
        # create zmesh
        zmesh = np.linspace(-h,d_height-h,res)

        # calculate adjustment length-scale of the canopy
        L_c = 1/(C*a)
        print("Adjustment length scale, L_c = {} m".format(L_c))
        print("h/L_c = {}. Model assumes h/L_c >> 1; verify that this is the case!!".format(h/L_c))

        # calculate canopy mixing length and print
        if beta is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                beta = u_star/U_h
                # U_h==0 and/or u_star==0 implies no flow
                beta[beta == np.inf] = 0
                beta[beta == -np.inf] = 0
                beta[beta == np.nan] = 0
        l = 2*beta**3*L_c
        print("Canopy mixing length, l = {} m".format(l))

        if u_star is not None:
            if U_h is not None:
                assert np.isclose(U_h*beta, u_star), "Flow not set: the relation U_h=u_star/beta must be satisfied."
            else:
                # calculate mean wind speed at top of the canopy
                with np.errstate(divide='ignore', invalid='ignore'):
                    U_h = u_star/beta
                    # beta==0 and/or u_star==0 implies no flow
                    U_h[U_h == np.inf] = 0
                    U_h[U_h == -np.inf] = 0
                    U_h[U_h == np.nan] = 0
                print("Mean wind spead at canopy top, U_h = {} {}/s".format(U_h, self.units))
        else:
            assert U_h is not None, "Flow not set: One of u_star or U_h must be specified."
            u_star = U_h*beta
            #print("Canopy friction velocity, u_star = {}.".format(U_h*beta))

        # calculate constants needed above canopy
        kappa = 0.4 # von Karman's constant
        d = l/kappa

        # calculate vertical wind profile at given resolution
        if iter_length is None:
            U_B = np.zeros_like(zmesh)
            U_B[zmesh<=0] = U_h*np.exp(beta*zmesh[zmesh<=0]/l)
            U_B[zmesh>0] = u_star/kappa*np.log((zmesh[zmesh>0]+d)/(d*np.exp(-kappa/beta)))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                beta_l = beta/l
                kappa_beta = kappa/beta
                if type(beta_l) is np.ndarray:
                    beta_l[beta_l == np.inf] = 0
                    kappa_beta[kappa_beta == np.inf] = 0
                    beta_l[beta_l == -np.inf] = 0
                    kappa_beta[kappa_beta == -np.inf] = 0
                    beta_l[beta_l == np.nan] = 0
                    kappa_beta[kappa_beta == np.nan] = 0
            U_B = np.zeros((iter_length,len(zmesh)))
            U_B[:,zmesh<=0] = np.tile(U_h,(len(zmesh[zmesh<=0]),1)).T*np.exp(
                            np.outer(beta_l,zmesh[zmesh<=0]))
            z_0_mat = np.tile(d*np.exp(-kappa_beta),(len(zmesh[zmesh>0]),1)).T
            U_B[:,zmesh>0] = np.tile(u_star/kappa,(len(zmesh[zmesh>0]),1)).T*np.log(
                            np.add.outer(d,zmesh[zmesh>0])/z_0_mat)
            # each row is a different time point, and z is across columns

        # broadcast to flow
        if iter_length is None:
            # time independent
            if len(self.L) == 2:
                # 2D
                flow = np.broadcast_to(U_B,(res,res))
                self.flow = [flow, np.zeros_like(flow)]
            else:
                # 3D
                flow = np.broadcast_to(U_B,(res,res,res))
                self.flow = [flow, np.zeros_like(flow), np.zeros_like(flow)]
            self.__set_flow_variables()
        else:
            # time dependent
            if len(self.L) == 2:
                # 2D
                # broadcast from (t,y) -> (x,t,y)
                flow = np.broadcast_to(U_B, (res, iter_length, res))
                # transpose: (x,t,y) -> (t,x,y)
                flow = np.transpose(flow, axes=(1, 0, 2))
                self.flow = [flow, np.zeros_like(flow)]
            else:
                # 3D
                # broadcast from (t,y) -> (x,y,t,z)
                flow = np.broadcast_to(U_B, (res, res, iter_length, res))
                # transpose: (x,y,t,z) -> (t,x,y,z)
                flow = np.transpose(flow, axes=(2,0,1,3))
                self.flow = [flow, np.zeros_like(flow), np.zeros_like(flow)]
            if tspan is None:
                self.__set_flow_variables(tspan=1)
            else:
                self.__set_flow_variables(tspan=tspan)


        # housekeeping
        self.__reset_flow_variables()
        self.fluid_domain_LLC = None
        self.h_p = h



    def read_IB2d_vtk_data(self, path, dt, print_dump, d_start=0, d_finish=None):
        '''Reads in vtk flow velocity data generated by IB2d and sets environment
        variables accordingly.

        Can read in vector data with filenames u.####.vtk or scalar data
        with filenames uX.####.vtk and uY.####.vtk.

        IB2d is an Immersed Boundary (IB) code for solving fully coupled
        fluid-structure interaction models in Python and MATLAB. The code is 
        hosted at https://github.com/nickabattista/IB2d

        Parameters
        ----------
        path : str
            path to folder with vtk data
        dt : float
            dt in input2d
        print_dump : int
            print_dump in input2d
        d_start : int, default=0
            number of first vtk dump to read in
        d_finish : int, optional
            number of last vtk dump to read in, or None to read to end
        '''

        ##### Parse parameters and read in data #####

        path = Path(path)
        if not path.is_dir(): 
            raise FileNotFoundError("Directory {} not found!".format(str(path)))

        #infer d_finish
        file_names = [x.name for x in path.iterdir() if x.is_file()]
        if 'u.' in [x[:2] for x in file_names]:
            u_nums = sorted([int(f[2:6]) for f in file_names if f[:2] == 'u.'])
            if d_finish is None:
                d_finish = u_nums[-1]
            vector_data = True
        else:
            assert 'uX.' in [x[:3] for x in file_names],\
                "Could not find u.####.vtk or uX.####.vtk files in {}.".format(str(path))
            u_nums = sorted([int(f[3:7]) for f in file_names if f[:3] == 'uX.'])
            if d_finish is None:
                d_finish = u_nums[-1]
            vector_data = False

        X_vel = []
        Y_vel = []
        path = str(path) # Get string for passing into functions

        print('Reading vtk data...')
        
        for n in range(d_start, d_finish+1):
            # Points to desired data viz_IB2d data file
            if n < 10:
                numSim = '000'+str(n)
            elif n < 100:
                numSim = '00'+str(n)
            elif n < 1000:
                numSim = '0'+str(n)
            else:
                numSim = str(n)

            # Imports (x,y) grid values and ALL Eulerian Data %
            #                      DEFINITIONS
            #          x: x-grid                y: y-grid
            #       Omega: vorticity           P: pressure
            #    uMag: mag. of velocity
            #    uX: mag. of x-Velocity   uY: mag. of y-Velocity
            #    u: velocity vector
            #    Fx: x-directed Force     Fy: y-directed Force
            #
            #  Note: U(j,i): j-corresponds to y-index, i to the x-index
            
            if vector_data:
                # read in vector velocity data
                strChoice = 'u'; xy = True
                uX, uY, x, y = dataio.read_2DEulerian_Data_From_vtk(path, numSim,
                                                                     strChoice,xy)
                X_vel.append(uX.T) # (y,x) -> (x,y) coordinates
                Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates
            else:
                # read in x-directed Velocity Magnitude #
                strChoice = 'uX'; xy = True
                uX,x,y = dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                               strChoice,xy)
                X_vel.append(uX.T) # (y,x) -> (x,y) coordinates

                # read in y-directed Velocity Magnitude #
                strChoice = 'uY'
                uY = dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                           strChoice)
                Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates

            ###### The following is just for reference! ######

            # read in Vorticity #
            # strChoice = 'Omega'; first = 0
            # Omega = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                               strChoice,first)
            # read in Pressure #
            # strChoice = 'P'; first = 0
            # P = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                           strChoice,first)
            # read in Velocity Magnitude #
            # strChoice = 'uMag'; first = 0
            # uMag = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                              strChoice,first)
            # read in x-directed Forces #
            # strChoice = 'Fx'; first = 0
            # Fx = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                            strChoice,first)
            # read in y-directed Forces #
            # strChoice = 'Fy'; first = 0
            # Fy = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                            strChoice,first)

            ###################################################

        print('Done!')

        ### Save data ###
        if d_start != d_finish:
            self.flow = [np.transpose(np.dstack(X_vel),(2,0,1)), 
                         np.transpose(np.dstack(Y_vel),(2,0,1))] 
            self.flow_times = np.arange(d_start,d_finish+1)*print_dump*dt
            # shift time so that flow starts at t=0
            self.flow_times -= self.flow_times[0]
        else:
            self.flow = [X_vel[0], Y_vel[0]]
            self.flow_times = None
        # shift domain to quadrant 1
        self.flow_points = (x-x[0], y-y[0])
        self.fluid_domain_LLC = (x[0], y[0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(2)]
        self.__reset_flow_variables()
        self.reset()



    def read_IBAMR3d_vtk_data(self, filename):
        '''Reads in vtk flow data from a single source and sets environment
        variables accordingly. The resulting flow will be time invarient. It is
        assumed this data is a rectilinear grid.

        All environment variables will be reset.

        Parameters
        ----------
        filename : string
            filename of data to read
        '''
        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))

        data, mesh, time = dataio.read_vtk_Rectilinear_Grid_Vector(filename)

        self.flow = list(data)
        self.flow_times = None

        # shift domain to quadrant 1
        self.flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                            mesh[2]-mesh[2][0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(3)]
        self.__reset_flow_variables()
        # record the original lower left corner (can be useful for later imports)
        self.fluid_domain_LLC = (mesh[0][0], mesh[1][0], mesh[2][0])
        # reset time
        self.reset()



    def read_IBAMR3d_vtk_dataset(self, path, start=None, finish=None):
        '''Reads in vtk flow data generated by VisIt from IBAMR and sets
        environment variables accordingly. Assumes that the vtk filenames are
        IBAMR_db_###.vtk where ### is the dump number, as automatically done
        with read_IBAMR3d_py27.py. Also assumes that the mesh is the same
        for each vtk.

        Imported times will be translated backward so that the first time loaded
        corresponds to an agent environment time of 0.0.

        Parameters
        ----------
        path : string
            path to vtk data
        start : int, optional
            vtk file number to start with. If None, start at first one.
        finish : int, optional
            vtk file number to end with. If None, end with last one.
        '''

        path = Path(path)
        if not path.is_dir(): 
            raise FileNotFoundError("Directory {} not found!".format(str(path)))
        file_names = [x.name for x in path.iterdir() if x.is_file() and
                      x.name[:9] == 'IBAMR_db_']
        file_nums = sorted([int(f[9:12]) for f in file_names])
        if start is None:
            start = file_nums[0]
        else:
            start = int(start)
            assert start in file_nums, "Start number not found!"
        if finish is None:
            finish = file_nums[-1]
        else:
            finish = int(finish)
            assert finish in file_nums, "Finish number not found!"

        ### Gather data ###
        flow = [[], [], []]
        flow_times = []

        for n in range(start, finish+1):
            if n < 10:
                num = '00'+str(n)
            elif n < 100:
                num = '0'+str(n)
            else:
                num = str(n)
            this_file = path / ('IBAMR_db_'+num+'.vtk')
            data, mesh, time = dataio.read_vtk_Rectilinear_Grid_Vector(str(this_file))
            for dim in range(3):
                flow[dim].append(data[dim])
            flow_times.append(time)

        ### Save data ###
        self.flow = [np.array(flow[0]).squeeze(), np.array(flow[1]).squeeze(),
                     np.array(flow[2]).squeeze()]
        # parse time information
        if None not in flow_times and len(flow_times) > 1:
            # shift time so that the first time is 0.
            self.flow_times = np.array(flow_times) - min(flow_times)
        elif None in flow_times and len(flow_times) > 1:
            # could not parse time information
            warnings.warn("Could not retrieve time information from at least"+
                          " one vtk file. Assuming unit time-steps...", UserWarning)
            self.flow_times = np.arange(len(flow_times))
        else:
            self.flow_times = None
        # shift domain to quadrant 1
        self.flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                            mesh[2]-mesh[2][0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(3)]
        self.__reset_flow_variables()
        # record the original lower left corner (can be useful for later imports)
        self.fluid_domain_LLC = (mesh[0][0], mesh[1][0], mesh[2][0])
        # reset time
        self.reset()



    def read_comsol_vtu_data(self, filename, vel_conv=None, grid_conv=None):
        '''Reads in vtu flow data from a single source and sets environment
        variables accordingly. The resulting flow will be time invarient.
        It is assumed this data is on a regular grid and that a grid section
        is included in the data.

        FOR NOW, THIS IS TIME INVARIANT ONLY.

        All environment variables will be reset.

        Parameters
        ----------
        filename : string
            filename of data to read, incl. file extension
        vel_conv : float, optional
            scalar to multiply the velocity by in order to convert units
        grid_conv : float, optional
            scalar to multiply the grid by in order to convert units
        '''
        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(str(filename)))

        data, mesh = dataio.read_vtu_mesh_velocity(filename)

        if vel_conv is not None:
            print("Converting vel units by a factor of {}.".format(vel_conv))
            for ii, d in enumerate(data):
                data[ii] = d*vel_conv
        if grid_conv is not None:
            print("Converting grid units by a factor of {}.".format(grid_conv))
            for ii, m in enumerate(mesh):
                mesh[ii] = m*grid_conv

        self.flow = data
        self.flow_times = None

        # shift domain to quadrant 1
        self.flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                            mesh[2]-mesh[2][0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(3)]
        self.__reset_flow_variables()
        # record the original lower left corner (can be useful for later imports)
        self.fluid_domain_LLC = (mesh[0][0], mesh[1][0], mesh[2][0])
        # reset time
        self.reset()



    def read_stl_mesh_data(self, filename):
        '''Reads in 3D mesh data from an ascii or binary stl file. Must have
        the numpy-stl library installed. It is assumed that the coordinate
        system of the stl mesh matches the coordinate system of the flow field.
        Thus, the mesh will be translated using the flow LLC if necessary.'''

        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))

        ibmesh, self.max_meshpt_dist = dataio.read_stl_mesh(filename)

        # shift coordinates to match any shift that happened in flow data
        if self.fluid_domain_LLC is not None:
            for ii in range(3):
                ibmesh[:,:,ii] -= self.fluid_domain_LLC[ii]
        
        self.ibmesh = ibmesh



    def read_IB2d_vertex_data(self, filename, res_factor=0.501):
        '''Reads in 2D vertex data from a .vertex file (IB2d). Assumes that any 
        vertices closer than res_factor (default is half + a bit for numerical 
        stability) times the Eulerian mesh resolution are connected linearly. 
        Thus, the flow data must be imported first!
        '''

        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))
        assert self.flow_points is not None, "Must import flow data first!"
        dists = np.concatenate([self.flow_points[ii][1:]-self.flow_points[ii][0:-1]
                                for ii in range(2)])
        Eulerian_res = dists.min()

        vertices = dataio.read_IB2d_vertices(filename)
        print("Processing vertex file for point-wise connections within {}.".format(
            res_factor*Eulerian_res))
        dist_mat_test = distance.pdist(vertices)<=res_factor*Eulerian_res
        idx = np.array(list(combinations(range(vertices.shape[0]),2)))
        self.ibmesh = np.array([vertices[idx[dist_mat_test,0],:],
                                vertices[idx[dist_mat_test,1],:]])
        self.ibmesh = np.transpose(self.ibmesh,(1,0,2))
        print("Done! Visually check structure with environment.plot_envir().")
        # shift coordinates to match any shift that happened in flow data
        if self.fluid_domain_LLC is not None:
            for ii in range(2):
                self.ibmesh[:,:,ii] -= self.fluid_domain_LLC[ii]
        self.max_meshpt_dist = np.linalg.norm(self.ibmesh[:,0,:]-self.ibmesh[:,1,:],axis=1).max()



    def add_vertices_to_2D_ibmesh(self):
        '''Methods for auto-connecting mesh vertices into line segments may
        result in line segments that cross each other away from vertex points.
        This will cause undesirable behavior including mesh crossing. This
        method adds a vertex point at any such intersection to avoid such
        problems.'''

        if self.ibmesh is None:
            print("No ibmesh found!")
            return
        elif self.ibmesh.shape[1] > 2:
            print("This method does not work for 3D problems!")
            return

        new_ibmesh = [seg for seg in self.ibmesh]

        # we will be altering the mesh as we go, so loop until we have considered
        #   all segments once.
        n = 0
        print("Breaking up intersecting mesh segments...")
        # go until there are no longer at least two segments to consider
        while n+1 < len(new_ibmesh):
            seg = new_ibmesh[n]
            forward_meshes = np.array(new_ibmesh[n+1:])
            # Find all mesh elements that have points within max_meshpt_dist*2
            pt_bool = np.linalg.norm(
                forward_meshes.reshape((forward_meshes.shape[0]*forward_meshes.shape[1],
                                        forward_meshes.shape[2]))-seg[0,:],
                axis=1)<self.max_meshpt_dist*2
            pt_bool = pt_bool.reshape((forward_meshes.shape[0],forward_meshes.shape[1]))
            close_mesh = forward_meshes[np.any(pt_bool,axis=1)]
            intersections = planktos.swarm._seg_intersect_2D(seg[0,:], seg[1,:], 
                close_mesh[:,0,:], close_mesh[:,1,:], get_all=True)
            if intersections is None:
                # Nothing to do; increment counter and continue
                n += 1
                continue
            else:
                # Use s_I to order the intersections and then go down the line,
                #   breaking segments into new segments.
                intersections = sorted(intersections, key=lambda x: x[1])
                new_seg_pieces = []
                new_int_pieces = []
                for intersection in intersections:
                    if len(new_seg_pieces) == 0:
                        # first one has P0 as an endpoint
                        new_seg_pieces.append(
                            np.array([seg[0,:],intersection[0]]))
                    else:
                        # each other has the second point of the previous segment
                        #   as an endpoint
                        new_seg_pieces.append(
                            np.array([new_seg_pieces[-1][1,:],intersection[0]]))
                    # now break up the segment that intersected seg
                    new_int_pieces.append(
                        np.array([intersection[3],intersection[0]]))
                    new_int_pieces.append(
                        np.array([intersection[0],intersection[4]]))
                    # and delete the original intersecting segment from the list
                    for ii, elem in enumerate(new_ibmesh[n+1:]):
                        if np.all(elem == np.array([intersection[3],intersection[4]]))\
                            or np.all(elem == np.array([intersection[4],intersection[3]])):
                            new_ibmesh.pop(ii+n+1)
                            break
                # add on the last segment which ends at the other endpoint
                new_seg_pieces.append(
                    np.array([new_seg_pieces[-1][1,:],seg[1,:]]))
                # remove the mesh piece we just examined and replace with the
                #   broken up ones
                new_ibmesh.pop(n)
                new_ibmesh[n:n] = new_seg_pieces
                # add the sliced up intersecting elements to the end
                new_ibmesh += new_int_pieces
                # increment counter by the number of seg pieces we added
                n += len(new_seg_pieces)
        # Done.

        # Replace ibmesh with new_ibmesh
        self.ibmesh =  np.array(new_ibmesh)



    def read_vertex_data_to_convex_hull(self, filename):
        '''Reads in 2D or 3D vertex data from a vtk file or a vertex file and 
        applies ConvexHull triangulation to get a complete boundary. This uses 
        Qhull through Scipy under the hood http://www.qhull.org/.
        '''

        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))

        if filename.strip()[-4:] == '.vtk':
            points, bounds = dataio.read_vtk_Unstructured_Grid_Points(filename)
        elif filename.strip()[-7:] == '.vertex':
            points = dataio.read_IB2d_vertices(filename)
        else:
            raise RuntimeError("File extension for {} not recognized.".format(filename))

        # get dimension
        DIM = points.shape[1]

        # shift to first quadrant

        hull = ConvexHull(points)
        if self.fluid_domain_LLC is not None:
            for ii in range(DIM):
                points[:,ii] -= self.fluid_domain_LLC[ii]
        self.ibmesh = points[hull.simplices]
        max_len = np.concatenate(tuple(
                  np.linalg.norm(self.ibmesh[:,ii,:]-self.ibmesh[:,(ii+1)%DIM,:], axis=1)
                  for ii  in range(DIM)
                  )).max()
        self.max_meshpt_dist = max_len



    def tile_flow(self, x=2, y=1):
        '''Tile fluid flow a number of times in the x and/or y directions.
        While obviously this works best if the fluid is periodic in the
        direction(s) being tiled, this will not be enforced. Instead, it will
        just be assumed that the domain edges are equivalent, and only the
        right/upper domain edge will be used in tiling.

        Parameters
        ----------
        x : int, default=2
            number of tiles in the x direction (counting the one already there)
        y : int, default=1
            number of tiles in the y direction (counting the one already there)
        '''

        DIM3 = len(self.L) == 3
        TIME_DEP = len(self.flow[0].shape) != len(self.L)

        if x is None:
            x = 1
        if y is None:
            y = 1
        if DIM3:
            tile_num = (x,y,1)
        else:
            tile_num = (x,y)

        if not TIME_DEP:
            # no time dependence
            flow_shape = self.flow[0].shape
            new_flow_shape = np.array(flow_shape)
            # get new dimensions
            for dim in range(len(flow_shape)):
                new_flow_shape[dim] += (flow_shape[dim]-1)*(tile_num[dim]-1)

            for n, flow in enumerate(self.flow):
                new_flow = np.zeros(new_flow_shape)
                # copy bottom-left corner
                new_flow[0,0,...] = flow[0,0,...]
                # tile first row/column
                if DIM3:
                    r_tile_num = (x,1)
                    c_tile_num = (y,1)
                else:
                    r_tile_num = x
                    c_tile_num = y
                new_flow[1:,0,...] = np.tile(flow[1:,0,...], r_tile_num)
                new_flow[0,1:,...] = np.tile(flow[0,1:,...], c_tile_num)
                # tile interior
                new_flow[1:,1:,...] = np.tile(flow[1:,1:,...], tile_num)
                self.flow[n] = new_flow
        else:
            # time dependent flow
            tile_num_time = [1]+list(tile_num) # prepend a 1; do not tile time
            flow_shape = self.flow[0].shape
            new_flow_shape = np.array(flow_shape)
            # get new dimensions
            for dim in range(1,len(flow_shape)):
                new_flow_shape[dim] += (flow_shape[dim]-1)*(tile_num_time[dim]-1)

            for n, flow in enumerate(self.flow):
                new_flow = np.zeros(new_flow_shape)
                # copy bottom-left corner
                new_flow[:,0,0,...] = flow[:,0,0,...]
                # tile first row/column
                if DIM3:
                    r_tile_num = (1,x,1)
                    c_tile_num = (1,y,1)
                else:
                    r_tile_num = (1,x)
                    c_tile_num = (1,y)
                new_flow[:,1:,0,...] = np.tile(flow[:,1:,0,...], r_tile_num)
                new_flow[:,0,1:,...] = np.tile(flow[:,0,1:,...], c_tile_num)
                # tile interior
                new_flow[:,1:,1:,...] = np.tile(flow[:,1:,1:,...], tile_num_time)
                self.flow[n] = new_flow
            
        # update environment dimensions and Eulerian meshes
        new_points = []
        self.orig_L = tuple(self.L[:2])
        for n in range(2):
            self.L[n] *= tile_num[n]
            new_points.append(np.concatenate([self.flow_points[n]]+
                [x*self.flow_points[n][-1]+self.flow_points[n][1:] for
                x in range(1,tile_num[n])]))
        if DIM3:
            new_points.append(self.flow_points[2])
        self.flow_points = tuple(new_points)
        self.tiling = (x,y)

        # tile Lagrangian meshes
        if self.ibmesh is not None:
            newmeshs = [self.ibmesh]
            for ii in range(self.tiling[0]):
                for jj in range(self.tiling[1]):
                    new_mesh = np.array(self.ibmesh)
                    new_mesh[:,:,0] += self.orig_L[0]*ii
                    new_mesh[:,:,1] += self.orig_L[1]*jj
                    newmeshs.append(new_mesh)
            self.ibmesh = np.concatenate(newmeshs)
        self.__reset_flow_deriv()



    def extend(self, x_minus=0, x_plus=0, y_minus=0, y_plus=0):
        '''Duplicate the boundary of the fluid flow a number of times in 
        the x (+ or -) and/or y (+ or -) directions, thus extending the domain
        with constant fluid velocity. Good for extending domains with resolved
        fluid flow before/after and on the sides of a structure.

        Parameters
        ----------
        x_minus : int
            number of times to duplicate bndry in the x- direction
        x_plus : int
            number of times to duplicate bndry in the x+ direction
        y_minus : int
            number of times to duplicate bndry in the y- direction
        y_plus : int
            number of times to duplicate bndry in the y+ direction
        '''

        DIM3 = len(self.L) == 3
        TIME_DEP = len(self.flow[0].shape) != len(self.L)

        assert x_minus>=0 and x_plus>=0 and y_minus>=0 and y_plus>=0,\
            "arguments must be nonnegative"

        if not TIME_DEP:
            res_x = self.L[0]/(self.flow[0].shape[0]-1)
            res_y = self.L[1]/(self.flow[0].shape[1]-1)
            for dim in range(len(self.L)):
                # first, extend in x direction
                self.flow[dim] = np.concatenate(tuple(
                    np.array([self.flow[dim][0,...]]) for ii in range(x_minus)
                    ) + (self.flow[dim],) + tuple(
                    np.array([self.flow[dim][-1,...]]) for jj in range(x_plus)
                    ), axis=0)
                # next, extend in y direction. This requires a shuffling
                #   (tranpose) of dimensions...
                if DIM3:
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,0,:]]).transpose(1,0,2)
                        for ii in range(y_minus)
                        ) + (self.flow[dim],) + tuple(
                        np.array([self.flow[dim][:,-1,:]]).transpose(1,0,2)
                        for jj in range(y_plus)
                        ), axis=1)
                else:
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,0]]).T
                        for ii in range(y_minus)
                        ) + (self.flow[dim],) + tuple(
                        np.array([self.flow[dim][:,-1]]).T
                        for jj in range(y_plus)
                        ), axis=1)

        else:
            res_x = self.L[0]/(self.flow[0].shape[1]-1)
            res_y = self.L[1]/(self.flow[0].shape[2]-1)
            for dim in range(len(self.L)):
                if DIM3:
                    # first, extend in x direction
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,0,...]]).transpose(1,0,2,3) 
                        for ii in range(x_minus)
                        ) + (self.flow[dim],) + tuple(
                        np.array([self.flow[dim][:,-1,...]]).transpose(1,0,2,3) 
                        for jj in range(x_plus)
                        ), axis=1)
                    # next, extend in y direction
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,:,0,:]]).transpose(1,2,0,3)
                        for ii in range(y_minus)
                        ) + (self.flow[dim],) + tuple(
                        np.array([self.flow[dim][:,:,-1,:]]).transpose(1,2,0,3)
                        for jj in range(y_plus)
                        ), axis=2)
                else:
                    # first, extend in x direction
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,0,:]]).transpose(1,0,2) 
                        for ii in range(x_minus)
                        ) + (self.flow[dim],) + tuple(
                        np.array([self.flow[dim][:,-1,:]]).transpose(1,0,2) 
                        for jj in range(x_plus)
                        ), axis=1)
                    # next, extend in y direction
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,:,0]]).tranpsose(1,2,0)
                        for ii in range(y_minus)
                        ) + (self.flow[dim],) + tuple(
                        np.array([self.flow[dim][:,:,-1]]).transpose(1,2,0)
                        for jj in range(y_plus)
                        ), axis=2)

        # update environment dimensions and meshes
        new_points = []
        self.orig_L = tuple(self.L[:2])
        self.L[0] += res_x*(x_minus+x_plus)
        self.L[1] += res_y*(y_minus+y_plus)
        if x_minus+x_plus > 0:
            new_points.append(np.concatenate([self.flow_points[0]]+
                [self.flow_points[0][-1]+res_x*np.arange(1,x_minus+x_plus+1)]))
        else:
            new_points.append(self.flow_points[0])
        if y_minus+y_plus > 0:
            new_points.append(np.concatenate([self.flow_points[1]]+
                [self.flow_points[1][-1]+res_y*np.arange(1,y_minus+y_plus+1)]))
        else:
            new_points.append(self.flow_points[1])
        if DIM3:
            new_points.append(self.flow_points[2])
        self.flow_points = tuple(new_points)
        self.__reset_flow_deriv()



    def add_swarm(self, swarm_size=100, **kwargs):
        ''' Adds a swarm into this environment.

        Parameters
        ----------
        swarm_size : swarm object or int (size of swarm), default=100
            If a swarm object is given, all following parameters will be ignored 
            (since the object is already initialized)
        init : string
            Method for initializing positions. See swarm class for options.
        seed : int
            Seed for random number generator
        kwargs
            keyword arguments to be set as swarm properties (see swarm class for 
            details)
        '''

        if isinstance(swarm_size, planktos.swarm):
            swarm_size._change_envir(self)
        else:
            return planktos.swarm(swarm_size, self, **kwargs)
            


    def move_swarms(self, dt=1.0, params=None):
        '''Move all swarms in the environment'''

        for s in self.swarms:
            s.move(dt, params, update_time=False)

        # update time
        self.time_history.append(self.time)
        self.time += dt
        print('time = {}'.format(np.round(self.time,11)))



    def set_boundary_conditions(self, x_bndry, y_bndry, z_bndry=None):
        '''Check and set boundary conditions. Set Z-dimension only if
        zdim is not None. Each boundary condition must be either a list or an
        iterable of length 2.
        '''

        supprted_conds = ['zero', 'noflux']
        default_conds_x = ('zero', 'zero')
        default_conds_y = ('zero', 'zero')
        default_conds_z = ('noflux', 'noflux')

        self.bndry = []

        if x_bndry is None:
            # default boundary conditions
            self.bndry.append(default_conds_x)
        elif isinstance(x_bndry, str):
            if x_bndry not in supprted_conds:
                self.bndry = [default_conds_x, default_conds_y, default_conds_z]
                raise NameError("X boundary condition {} not implemented.".format(x_bndry))
            self.bndry.append([x_bndry, x_bndry])
        else:
            try:
                iter(x_bndry)
            except TypeError:
                print("x_bndry must be either an iterable or a string.")
                self.bndry = [default_conds_x, default_conds_y, default_conds_z]
                raise
            else:
                if x_bndry[0] not in supprted_conds or x_bndry[1] not in supprted_conds:
                    self.bndry = [default_conds_x, default_conds_y, default_conds_z]
                    raise NameError("x boundary condition {} not implemented.".format(x_bndry))
                else:
                    self.bndry.append(x_bndry)

        if y_bndry is None:
            # default boundary conditions
            self.bndry.append(default_conds_y)
        elif isinstance(y_bndry, str):
            if y_bndry not in supprted_conds:
                self.bndry += [default_conds_y, default_conds_z]
                raise NameError("Y boundary condition {} not implemented.".format(y_bndry))
            self.bndry.append([y_bndry, y_bndry])
        else:
            try:
                iter(y_bndry)
            except TypeError:
                print("y_bndry must be either an iterable or a string.")
                self.bndry += [default_conds_y, default_conds_z]
                raise
            else:
                if y_bndry[0] not in supprted_conds or y_bndry[1] not in supprted_conds:
                    self.bndry += [default_conds_y, default_conds_z]
                    raise NameError("y boundary condition {} not implemented.".format(y_bndry))
                else:
                    self.bndry.append(y_bndry)

        if z_bndry is None:
            # default boundary conditions
            self.bndry.append(default_conds_z)
        elif isinstance(z_bndry, str):
            if z_bndry not in supprted_conds:
                self.bndry.append(default_conds_z)
                raise NameError("Z boundary condition {} not implemented.".format(z_bndry))
            self.bndry.append([z_bndry, z_bndry])
        else:
            try:
                iter(z_bndry)
            except TypeError:
                self.bndry.append(default_conds_z)
                print("z_bndry must be either an iterable or a string.")
                raise
            else:
                if z_bndry[0] not in supprted_conds or z_bndry[1] not in supprted_conds:
                    self.bndry.append(default_conds_z)
                    raise NameError("z boundary condition {} not implemented.".format(z_bndry))
                else:
                    self.bndry.append(z_bndry)



    def interpolate_temporal_flow(self, t_indx=None, time=None):
        '''Interpolate flow in time using a cubic spline. Defaults to 
        interpolating at the current time, given by self.time.

        Parameters
        ----------
        t_indx : int
            Interpolate at a time referred to by self.envir.time_history[t_indx]
        time : float
            Interpolate at a specific time

        Returns
        -------
            interpolated flow field as a list of ndarrays
        '''

        # If PPoly CubicSplines do not exist, create them.
        if self.t_interp is None:
            self._create_temporal_interpolations()

        if t_indx is None and time is None:
            time = self.time
        elif t_indx is not None:
            time = self.time_history[t_indx]

        # Enforce constant extrapolation
        if time <= self.flow_times[0]:
            return [f[0, ...] for f in self.flow]
        elif time >= self.flow_times[-1]:
            return [f[-1, ...] for f in self.flow]
        else:
            # interpolate
            return [f(time) for f in self.t_interp]



    def _create_temporal_interpolations(self):
        '''Create PPoly CubicSplines to interpolate the fluid velocity in time.'''
        self.t_interp = []
        self.dt_interp = []
        for flow in self.flow:
            # Defaults to axis=0 along which data is varying, which is t axis
            # Defaults to not-a-knot boundary condition, resulting in first
            #   and second segments at curve ends being the same polynomial
            # Defaults to extrapolating out-of-bounds points based on first
            #   and last intervals. This will be overriden by this method
            #   to use constant extrapolation instead.
            self.t_interp.append(interpolate.CubicSpline(self.flow_times, flow))
            self.dt_interp.append(self.t_interp[-1].derivative())



    def interpolate_flow(self, positions, flow=None, time=None, method='linear'):
        '''Spatially interpolate the fluid velocity field (or another flow field) 
        at the supplied positions. If flow is None and self.flow is time-varying,
        the flow field will be interpolated in time first, using the current 
        environmental time, or a different time if provided.

        Parameters
        ----------
        positions : array
            NxD locations at which to interpolate the flow field, where D is the 
            dimension of the system.
        flow : list of arrays, optional
            if None, the environmental flow field. interpolated in time if 
            necessary.
        time : float, optional
            if None, the present time. Otherwise, the flow field will be
            interpolated to the time given.
        method : string, default='linear'
            spatial interpolation method to be passed to 
            scipy.interpolate.interpn. Anything but splinef2d is supported.

        '''

        if flow is None:
            if len(self.flow[0].shape) == len(self.L):
                # non time-varying flow
                flow = self.flow
            else:
                flow = self.interpolate_temporal_flow(time=time)
        
        if method == 'splinef2d':
            raise RuntimeError('Extrapolation is not supported in splinef2d.'+
                               ' This is needed for RK4 solvers, and so is not'+
                               ' a supported method in interpolate_flow.')

        x_vel = interpolate.interpn(self.flow_points, flow[0],
                                    positions, method=method, 
                                    bounds_error=False, fill_value=None)
        y_vel = interpolate.interpn(self.flow_points, flow[1],
                                    positions, method=method, 
                                    bounds_error=False, fill_value=None)
        if len(flow) == 3:
            z_vel = interpolate.interpn(self.flow_points, flow[2],
                                        positions, method=method, 
                                        bounds_error=False, fill_value=None)
            return np.array([x_vel, y_vel, z_vel]).T
        else:
            return np.array([x_vel, y_vel]).T



    def get_mean_fluid_speed(self):
        '''Return the mean fluid speed at the current time, temporally
        interpolating the flow field if necessary.'''

        DIM3 = (len(self.L) == 3)

        if (not DIM3 and len(self.flow[0].shape) == 2) or \
            (DIM3 and len(self.flow[0].shape) == 3):
            # temporally constant flow
            flow_now = self.flow
        else:
            # temporal flow. interpolate in time, and then in space.
            flow_now = self.interpolate_temporal_flow()

        if DIM3:
            fluid_speed = np.sqrt(flow_now[0]**2+flow_now[1]**2+flow_now[2]**2)
        else:
            fluid_speed = np.sqrt(flow_now[0]**2+flow_now[1]**2)

        return np.mean(fluid_speed)



    def calculate_mag_gradient(self):
        '''Calculate and store the gradient of the magnitude of the fluid velocity 
        at the current time, along with the time at which it was calculated. 
        Gradient is calculated via second order accurate central differences 
        (using numpy) with second order accuracy at the boundaries and saved in 
        case it is needed again.'''

        TIME_DEP = len(self.flow[0].shape) != len(self.L)
        if not TIME_DEP:
            flow_grad = np.gradient(np.sqrt(
                            np.sum(np.array(self.flow)**2, axis=0)
                            ), *self.flow_points, edge_order=2)
        else:
            # first, interpolate flow in time. Then calculate gradient.
            flow_grad = np.gradient(
                            np.sqrt(np.sum(
                            np.array(self.interpolate_temporal_flow())**2,
                            axis=0)), *self.flow_points, edge_order=2)
        # save the newly calculuate gradient
        self.grad = flow_grad
        self.grad_time = self.time



    def calculate_FTLE(self, grid_dim=None, testdir=None, t0=0, T=0.1, dt=0.001, 
                       ode_gen=None, props=None, t_bound=None, swrm=None, 
                       params=None):
        '''Calculate the FTLE field at the given time(s) t0 with integration 
        length T on a discrete grid with given dimensions. The calculation will 
        be conducted with respect to the fluid velocity field loaded in this 
        environment and either tracer particle movement (default), an ode specifying
        deterministic equations of motion, or other arbitrary particle movement 
        as specified by a swarm object's get_positions method and updated in 
        discrete time intervals of length dt.

        All FTLE calculations will be done using a swarm object. This means that:
        
        1) The boundary conditions specified by this environment will be respected.
        2) Immersed boundaries (if any are loaded into this environment) will be 
           treated as impassible to all particles and movement vectors crossing these 
           boundaries will be projected onto them.

        If passing in a set of ode or finding the FTLE field for tracer particles, 
        an RK45 solver will be used. Otherwise, integration will be via the 
        swarm object's get_positions method.

        If both ode and swarm arguments are None, the default is to calculate the 
        FTLE based on massless tracer particles.

        Parameters
        ----------
        grid_dim : tuple of int 
            size of the grid in each dimension (x, y, [z]). Defaults to the 
            fluid grid.
        testdir : str
            grid points can heuristically be removed from the interior of 
            immersed structures. To accomplish this, a line will be drawn from 
            each point to a domain boundary. If the number of intersections
            is odd, the point is considered interior and masked. See grid_init 
            for details - this argument sets the direction that the line 
            will be drawn in (e.g. 'x1' for positive x-direction, 'y0' for 
            negative y-direction). If None, do not perform this check and use 
            all gridpoints.
        t0 : float, optional
            start time for calculating FTLE. If None, default 
            behavior is to set t0=0.
            TODO: Interable to calculate at many times. Default then becomes 
            t0=0 for time invariant flows and calculate FTLE at all times 
            the flow field was specified at (self.flow_times) 
            for time varying flows?
        T : float, default=0.1
            integration time. Default is 1, but longer is better (up to a point).
        dt : float, default=0.001
            if solving ode or tracer particles, this is the time step for 
            checking boundary conditions. If passing in a swarm object, 
            this argument represents the length of the Euler time steps.
        ode_gen : function handle, optional
            functional handle for an ode generator that takes
            in a swarm object and returns an ode function handle with
            call signature ODEs(t,x), where t is the current time (float) 
            and x is a 2*NxD array with the first N rows giving v=dxdt and 
            the second N rows giving dvdt. D is the spatial dimension of 
            the problem. See the ODE generator functions in motion.py for 
            examples of format. 
            The ODEs will be solved using RK45 with a newly created swarm 
            specified on a grid throughout the domain.
        props : dict, optional 
            dictionary of properties for the swarm that will be created to solve 
            the odes. Effectively, this passes parameter values into the ode 
            generator.
        t_bound : float, optional
            if solving ode or tracer particles, this is the bound on
            the RK45 integration step size. Defaults to dt/100.
        swarm : swarm object, optional 
            swarm object with user-defined movement rules as 
            specified by the get_positions method. This allows for arbitrary 
            FTLE calculations through subclassing and overriding this method. 
            Steps of length dt will be taken until the integration length T 
            is reached. The swarm object itself will not be altered; a shallow 
            copy will be created for the purpose of calculating the FTLE on 
            a grid.
        params : dict, optional 
            params to be passed to supplied swarm object's get_positions method.

        Returns
        -------
        swarm object
            used to calculuate the FTLE
        list
            list of dt integration steps
        ndarray
            array the same size as the point grid giving the last time
            in the integration before each point exited the domain
        '''

        ###########################################################
        ######              Setup swarm object               ######
        ###########################################################
        if grid_dim is None:
            grid_dim = tuple(len(pts) for pts in self.flow_points)

        if len(grid_dim) == 3:
            print("Warning: FTLE has not been well tested for 3D cases!")

        if swrm is None:
            s = planktos.swarm(envir=self, shared_props=props, init='grid', 
                      grid_dim=grid_dim, testdir=testdir)
            # NOTE: swarm has been appended to this environment!
        else:
            # Get a soft copy of the swarm passed in
            s = copy.copy(swrm)
            # Add swarm to environment and re-initialize swarm positions
            self.add_swarm(s)
            s.positions = s.grid_init(*grid_dim, testdir=testdir)
            s.pos_history = []
            if self.flow is not None:
                s.velocities = ma.array(s.get_fluid_drift(), mask=s.positions.mask.copy())
            else:
                s.velocities = ma.array(np.zeros((s.positions.shape[0], len(self.L))), 
                                        mask=s.positions.mask.copy()) 
            s.accelerations = ma.array(np.zeros((s.positions.shape[0], len(self.L))),
                                        mask=s.positions.mask.copy())

        # get an array to record the corresponding last time for these positions
        last_time = ma.masked_array(np.ones(s.positions.shape[0])*t0, mask=s.positions[:,0].mask.copy())
        last_time.harden_mask()

        ###########################################################
        ######              Solve for positions              ######
        ###########################################################

        ###### OPTION A: Solve ODEs if no swarm object was passed in ######

        prnt_str = "Solving for positions from time {} to time {}:".format(t0,T)
        # NOTE: the scipy.integrate solvers convert masked arrays into ndarrays, 
        #   removing the mask entirely. We want to integrate only over the non-masked
        #   components for efficiency.
        if swrm is None:
            ### SETUP SOLVER ###
            if ode_gen is None:
                ode_fun = motion.tracer_particles(s, incl_dvdt=False)
                print("Finding {}D FTLE based on tracer particles.".format(len(grid_dim)))
            else:
                ode_fun = ode_gen(s)
                print("Finding {}D FTLE based on supplied ODE generator.".format(len(grid_dim)))
            print(prnt_str)

            # keep a list of all times solved for 
            #   (time history normally stored in environment class)
            current_time = t0
            time_list = [] 

            ### SOLVE ###
            while current_time < T:
                new_time = min(current_time + dt,T)
                ### TODO: REDO THIS SOLVER!!!!!!!!
                if ode_gen is None:
                    y = s.positions[~s.positions[:,0].mask,:]
                else:
                    y = np.concatenate((s.positions[~s.positions[:,0].mask,:], 
                                        s.velocities[~s.velocities[:,0].mask,:]))
                try:
                    # solve
                    y_new = motion.RK45(ode_fun, current_time, y, new_time, h_start=t_bound)
                except Exception as err:
                    print('RK45 solver returned an error at time {} with step_size {}.'.format(
                          current_time, dt))
                    raise

                # Put current position in the history (maybe only do this if something exits??)
                s.pos_history.append(s.positions.copy())
                # pull solution into swarm object's position/velocity attributes
                if ode_gen is None:
                    s.positions[~s.positions[:,0].mask,:] = y_new
                else:
                    N = round(y_new.shape[0]/2)
                    s.positions[~s.positions[:,0].mask,:] = y_new[:N,:]
                    s.velocities[~s.velocities[:,0].mask,:] = y_new[N:,:]
                # apply boundary conditions
                old_mask = s.positions.mask.copy()
                s.apply_boundary_conditions()
                # copy time to non-masked locations
                last_time[~s.positions[:,0].mask] = new_time
                
                # check if there were any boundary exits.
                #   if so, save this state. Also, always keep the first state
                #   with the grid information.
                if np.any(old_mask != s.positions.mask) or current_time == t0:
                    # if anybody left, record the previous time in the history
                    #   as the last moment before disappearance.
                    time_list.append(current_time)
                else:
                    # if nobody left the domain, get rid of the state from the 
                    #   position history to save space
                    s.pos_history.pop()

                # if all agents have left the domain, quit early
                if np.all(s.positions.mask):
                    print('FTLE solver quit early at time {} after all agents left the domain.'.format(current_time))
                    break

                # pass forward new variables
                current_time = new_time
                if current_time == T:
                    time_list.append(current_time)
                print('t={}'.format(current_time))

            # DONE SOLVING

        ###### OPTION B: Run get_positions on supplied swarm object ######

        else:
            print("Finding {}D FTLE based on supplied swarm object.".format(len(grid_dim)))
            print(prnt_str)
            # save this environment's time history
            envir_time = self.time
            envir_time_history = list(self.time_history)
            # now track this swarm's time
            self.time = t0
            self.time_history = []

            while self.time < T:
                # Put current position in the history
                s.pos_history.append(s.positions.copy())
                # Update positions
                s.positions[:,:] = s.get_positions(dt, params)
                # Update velocity and acceleration
                velocity = (s.positions - s.pos_history[-1])/dt
                s.accelerations[:,:] = (velocity - s.velocities)/dt
                s.velocities[:,:] = velocity
                # Apply boundary conditions.
                s.apply_boundary_conditions()
                # Update time
                self.time_history.append(self.time)
                self.time += dt
                # copy time to non-masked locations
                last_time[~s.positions[:,0].mask] = self.time

                print('t={}'.format(self.time))

                # if all agents have left the domain, quit early
                if np.all(s.positions.mask):
                    print('FTLE solver quit early at time {} after all agents left the domain.'.format(self.time))
                    break

            # DONE SOLVING

            # record and reset environment time
            time_list = list(self.time_history)
            time_list.append(self.time)
            self.time = envir_time
            self.time_history = envir_time_history

        ###########################################################
        ######         Calculate FTLE at each point          ######
        ###########################################################

        # a list of times solved for is given by time_list
        # all positions are recorded in s.pos_history with s.positions for last time
        # last times before leaving the domain given by last_time
        print('Calculating FTLE field...')

        ### COLLECT FACTS ABOUT GRID GEOMETRY ###

        dx = self.L[0]/(grid_dim[0]-1)
        dy = self.L[1]/(grid_dim[1]-1)
        if len(self.L) > 2:
            dz = self.L[2]/(grid_dim[2]-1)
            DIM = 3
        else:
            DIM = 2
        
        ### MASK ALL POINTS THAT EXITED IMMEDIATELY ###

        last_time[last_time==t0] = ma.masked
        # reshape for sanity's sake.
        last_time = np.reshape(last_time,grid_dim)

        ### INITIALIZE SOLUTION STRUCTURES ###

        FTLE_largest = ma.masked_array(np.zeros_like(last_time), mask=last_time.mask.copy())
        FTLE_largest.harden_mask()
        FTLE_largest.set_fill_value(0)
        FTLE_smallest = ma.masked_array(np.zeros_like(last_time), mask=last_time.mask.copy())
        FTLE_smallest.harden_mask()
        FTLE_smallest.set_fill_value(0)

        ### LOOP OVER ALL GRID POINTS ###

        grid_loc_iter = np.ndindex(last_time.shape)
        for grid_loc in grid_loc_iter:

            ### DEBUG ###
            # flat_loc = np.ravel_multi_index((grid_loc[0],grid_loc[1]),grid_dim)
            # start_pos = s.pos_history[0][flat_loc,:]
            # if 0.96 < start_pos[0] < 0.98 and 0.15 < start_pos[1] < 0.18:
            #     import pdb; pdb.set_trace()
            # if the central point is masked, skip this calculation.
            #  (inside a structure or non-repairable edge point)
            if last_time.mask[grid_loc]:
                continue

            ### LAYOUT STENCIL ###
            if DIM == 2:
                diff_list = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype=int)
            else:
                diff_list = np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]], dtype=int)

            # first, deal with edge of domain cases
            for dim, loc in enumerate(grid_loc):
                    if loc == 0:
                        diff_list[dim*2,:] *= 0
                    elif loc == grid_dim[dim]-1:
                        diff_list[dim*2+1,:] *= 0
            
            # check for masked neighbors
            neigh_list = np.array(grid_loc, dtype=int) + diff_list
            cont = False # if non-repairable, make true and continue to next pt
            for n in range(0,diff_list.shape[0],2):
                mask_list = last_time.mask[tuple(neigh_list[n:n+2].T)]
                # if both points are masked, mask this point and skip
                if np.all(mask_list):
                    FTLE_largest[grid_loc] = ma.masked
                    FTLE_smallest[grid_loc] = ma.masked
                    cont = True
                    break
                # if only one point is masked, switch difference calculation
                #   unless the central point is already being used, in which case skip
                if np.any(mask_list):
                    if np.any(diff_list[n:n+2].sum(axis=1) == 0):
                        FTLE_largest[grid_loc] = ma.masked
                        FTLE_smallest[grid_loc] = ma.masked
                        cont = True
                        break
                    else:
                        diff_list[n:n+2][mask_list] *= 0
            if cont:
                continue
            # reform neigh_list
            neigh_list = np.array(grid_loc, dtype=int) + diff_list

            ### GET TIME AND POSITION INFO ###

            # find the time before the first neighbor (or current loc) exited
            t_calc = last_time[tuple(neigh_list.T)].min()
            t_idx = time_list.index(t_calc)

            # get relevant position list
            if t_idx < len(s.pos_history):
                pos = s.pos_history[t_idx]
            else:
                pos = s.positions

            # get stencil spacing and flattened indices
            x_mult = abs((neigh_list[1,:]-neigh_list[0,:]).sum())
            y_mult = abs((neigh_list[3,:]-neigh_list[2,:]).sum())
            if DIM==2:
                flat_neigh = np.ravel_multi_index((neigh_list[:,0],neigh_list[:,1]),grid_dim)
            else:
                flat_neigh = np.ravel_multi_index((neigh_list[:,0],neigh_list[:,1],neigh_list[:,2]),grid_dim)
                z_mult = abs((neigh_list[5,:]-neigh_list[4,:]).sum())

            ### CENTRAL DIFFERENCE GRADIENT ###

            dXdx = (pos[flat_neigh[1],0]-pos[flat_neigh[0],0])/(x_mult*dx)
            dXdy = (pos[flat_neigh[3],0]-pos[flat_neigh[2],0])/(y_mult*dy)
            dYdx = (pos[flat_neigh[1],1]-pos[flat_neigh[0],1])/(x_mult*dx)
            dYdy = (pos[flat_neigh[3],1]-pos[flat_neigh[2],1])/(y_mult*dy)

            if DIM == 3:
                # calculate 3D central difference gradient
                dXdz = (pos[flat_neigh[5],0]-pos[flat_neigh[4],0])/(z_mult*dz)
                dYdz = (pos[flat_neigh[5],1]-pos[flat_neigh[4],1])/(z_mult*dz)
                dZdx = (pos[flat_neigh[1],2]-pos[flat_neigh[0],2])/(x_mult*dx)
                dZdy = (pos[flat_neigh[3],2]-pos[flat_neigh[2],2])/(y_mult*dy)
                dZdz = (pos[flat_neigh[5],2]-pos[flat_neigh[4],2])/(z_mult*dz)

                phi = np.array([[dXdx, dXdy, dXdz],
                                [dYdx, dYdy, dYdz],
                                [dZdx, dZdy, dZdz]])
            else:
                # form up 2D central difference gradient
                phi = np.array([[dXdx, dXdy],
                                [dYdx, dYdy]])

            ### CALCULATE FTLE ###

            w,_ = np.linalg.eigh(phi.T@phi)
            if w[-1] <= 0:
                FTLE_largest[grid_loc] = ma.masked
            else:
                FTLE_largest[grid_loc] = log(sqrt(w[-1]))/T
            if w[0] <= 0:
                FTLE_smallest[grid_loc] = ma.masked
            else:
                FTLE_smallest[grid_loc] = log(sqrt(w[0]))/T

        ###### Save and cleanup ######
        self.FTLE_largest = FTLE_largest
        self.FTLE_smallest = FTLE_smallest
        self.FTLE_loc = s.pos_history[0]
        self.FTLE_t0 = t0
        self.FTLE_T = T
        self.FTLE_grid_dim = grid_dim

        ###### Print stats ######
        print("Out of {} total points, {} exited the domain during integration.".format(
              last_time.size, np.sum(last_time<T)))

        return self.swarms.pop(), time_list, last_time



    def get_2D_vorticity(self, t_indx=None, time=None, t_n=None):
        '''Calculuate the vorticity of the fluid velocity field at a given time.

        If all time arguments are None but the flow is time-varying, the vorticity
        at the current time will be returned. If more than one time argument is
        specified, only the first will be used.

        Parameters
        ----------
        t_indx : int
            integer time index into self.envir.time_history[t_indx]
        time : float
            time
        t_n : int
            integer time index into self.flow_times[t_n]

        Returns
        -------
        vorticity as an ndarray

        '''
        assert len(self.L) == 2, "Fluid velocity field must be 2D!"
        if (t_indx is not None or time is not None or t_n is not None) and self.flow_times is None:
            warnings.warn("Warning: flow is time invarient but time arg passed into get_2D_vorticity.",
                          RuntimeWarning)

        if len(self.flow[0].shape) > len(self.L):
            grid_dim = self.flow[0].shape[1:]
            grid_loc_iter = np.ndindex(grid_dim)
            TIMEVAR = True
        else:
            grid_dim = self.flow[0].shape
            grid_loc_iter = np.ndindex(grid_dim)
            TIMEVAR = False

        if TIMEVAR:
            if t_indx is not None or time is not None:
                flow = self.interpolate_temporal_flow(t_indx=t_indx, time=time)
                v_x = flow[0]
                v_y = flow[1]
            elif t_n is not None:
                v_x = self.flow[0][t_n]
                v_y = self.flow[1][t_n]
            else:
                flow = self.interpolate_temporal_flow()
                v_x = flow[0]
                v_y = flow[1]
        else:
            v_x = self.flow[0]
            v_y = self.flow[1]

        dx = self.L[0]/(grid_dim[0]-1)
        dy = self.L[1]/(grid_dim[1]-1)

        vort = np.zeros_like(v_x)

        ### LOOP OVER ALL GRID POINTS ###
        for grid_loc in grid_loc_iter:
            diff_list = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype=int)
            # first, deal with edge of domain cases
            for dim, loc in enumerate(grid_loc):
                if loc == 0:
                    diff_list[dim*2,:] *= 0
                elif loc == grid_dim[dim]-1:
                    diff_list[dim*2+1,:] *= 0
            neigh_list = np.array(grid_loc, dtype=int) + diff_list
            # get stencil spacing
            x_mult = abs((neigh_list[1,:]-neigh_list[0,:]).sum())
            y_mult = abs((neigh_list[3,:]-neigh_list[2,:]).sum())
            # central differencing
            dvydx = (v_y[tuple(neigh_list[1,:])]-v_y[tuple(neigh_list[0,:])])/(x_mult*dx)
            dvxdy = (v_x[tuple(neigh_list[3,:])]-v_x[tuple(neigh_list[2,:])])/(y_mult*dy)
            # vorticity
            vort[grid_loc] = dvydx - dvxdy

        return vort



    def save_2D_vorticity(self, path, name, time_history=True, flow_times=False):
        '''Save the vorticity of a 2D fluid velocity field as one or more vtk 
        files (one for each time point).

        Parameters
        ----------
        path : string
            location to save file(s)
        name : string
            prefix name for file(s)
        time_history : bool
            if True, save vorticity data for each time step in the simulation 
            history. Only for time-varying fluid.
        flow_times : bool
            if True, save vorticity data for each time at which the fluid 
            velocity data is explicitly specified.
        '''

        if time_history:
            for cyc, time in enumerate(self.time_history):
                vort = self.get_2D_vorticity(t_indx=cyc)
                dataio.write_vtk_2D_uniform_grid_scalars(path, name, vort, self.L, cyc, time)
            cycle = len(self.time_history)
        else:
            cycle = None
        if flow_times:
            if time_history:
                out_name = 'omega'
            else:
                out_name = name
            for cyc, time in enumerate(self.flow_times):
                vort = self.get_2D_vorticity(t_n=cyc)
                dataio.write_vtk_2D_uniform_grid_scalars(path, out_name, vort, self.L, cyc, time)
        if time_history or not flow_times:
            vort = self.get_2D_vorticity(time=self.time)
            dataio.write_vtk_2D_uniform_grid_scalars(path, name, vort, self.L, cycle, self.time)



    def save_fluid(self, path, name, time_history=True, flow_times=False):
        '''Save the fluid velocity field as one or more vtk files (one for each 
        time point).

        Parameters
        ----------
        path : str
            directory in which to store the files. if it does not exist, it will 
            be created
        name : str
            prefix to put on filenames
        time_history : bool
            if True, save for each time step in the simulation history, 
            interpolating as needed. Only for time-varying fluid.
        flow_times : bool
            if True, save at each time for which the fluid velocity data is 
            explicitly specified.
        '''

        if time_history:
            for cyc, time in enumerate(self.time_history):
                flow = self.interpolate_temporal_flow(t_indx=cyc)
                dataio.write_vtk_uniform_grid_vectors(path, name, flow, self.L, cyc, time)
            cycle = len(self.time_history)
        else:
            cycle = None
        if flow_times:
            if time_history:
                out_name = 'U'
            else:
                out_name = name
            for cyc, time in enumerate(self.flow_times):
                flow = self.interpolate_temporal_flow(time=time)
                dataio.write_vtk_uniform_grid_vectors(path, out_name, flow, self.L, cyc, time)
        if time_history or not flow_times:
            flow = self.interpolate_temporal_flow(time=self.time)
            dataio.write_vtk_uniform_grid_vectors(path, name, flow, self.L, cycle, self.time)



    @property
    def Re(self):
        '''Return the Reynolds number at the current time. Must have set 
        self.U, self.char_L and self.nu. Reynolds number is U*char_L/nu.'''

        return self.U*self.char_L/self.nu



    def dudt(self, t_indx=None, time=None):
        '''Return the derivative of the fluid velocity with respect to time.
        Defaults to interpolating at the current time, given by self.time.

        Parameters
        ----------
        t_indx : int, optional
            Interpolate at a time referred to by self.envir.time_history[t_indx]
        time : float, optional
            Interpolate at a specific time. default is current time.

        Returns
        -------
        interpolated flow field as a list of ndarrays
        '''

        DIM3 = (len(self.L) == 3)

        if t_indx is None and time is None:
            time = self.time
        elif t_indx is not None:
            time = self.time_history[t_indx]

        if (not DIM3 and len(self.flow[0].shape) == 2) or \
            (DIM3 and len(self.flow[0].shape) == 3):
            # temporally constant flow
            return [np.zeros_like(f) for f in self.flow]
        else:
            # temporal flow.
            # If PPoly CubicSplines do not exist, create them.
            if self.t_interp is None:
                self._create_temporal_interpolations()
            return [dfdt(time) for dfdt in self.dt_interp]



    def reset(self, rm_swarms=False):
        '''Resets environment to time=0. Swarm history will be lost, and all
        swarms will maintain their last position and velocities. 
        If rm_swarms=True, remove all swarms.'''

        self.time = 0.0
        self.time_history = []
        if rm_swarms:
            self.swarms = []
        else:
            for sw in self.swarms:
                sw.pos_history = []



    def __reset_flow_variables(self, incl_rho_mu_U=False):
        '''To be used when the fluid flow changes. Resets all the helper
        parameters.'''

        self.h_p = None
        self.tiling = None
        self.orig_L = None
        self.plot_structs = []
        self.plot_structs_args = []
        self.__reset_flow_deriv()
        if incl_rho_mu_U:
            self.mu = None
            self.rho = None
            self.U = None



    def __reset_flow_deriv(self):
        '''Reset properties that are derived from the flow velocity itself.'''

        self.grad = None
        self.grad_time = None
        self.t_interp = None
        self.dt_interp = None



    def _plot_setup(self, fig, nohist=False):
        ''' Setup figures for plotting '''

        ########## 2D plot ##########
        if len(self.L) == 2:
            if nohist:
                ax = plt.axes(xlim=(0, self.L[0]), ylim=(0, self.L[1]))
            else:
                # chop up the axes to include histograms
                left, width = 0.1, 0.65
                bottom, height = 0.1, 0.65
                bottom_h = left_h = left + width + 0.02

                rect_scatter = [left, bottom, width, height]
                rect_histx = [left, bottom_h, width, 0.2]
                rect_histy = [left_h, bottom, 0.2, height]

                ax = plt.axes(rect_scatter, xlim=(0, self.L[0]), 
                            ylim=(0, self.L[1]))
                axHistx = plt.axes(rect_histx)
                axHisty = plt.axes(rect_histy)

                # no labels on histogram next to scatter plot
                nullfmt = NullFormatter()
                axHistx.xaxis.set_major_formatter(nullfmt)
                axHisty.yaxis.set_major_formatter(nullfmt)

                # set histogram limits/ticks
                axHistx.set_xlim(ax.get_xlim())
                axHisty.set_ylim(ax.get_ylim())
                int_ticks = MaxNLocator(nbins='auto', integer=True)
                pruned_ticks = MaxNLocator(prune='lower', nbins='auto',
                                        integer=True, min_n_ticks=3)
                axHistx.yaxis.set_major_locator(int_ticks)
                axHisty.xaxis.set_major_locator(pruned_ticks)

                # at this point we have a good looking image, but the aspect ratio
                #   is wrong. Adjust to make it right:
                old_ax_position = np.array(ax.get_position().get_points())
                ax.set_aspect('equal') # this should only shrink the plot?
                ax_position = ax.get_position().get_points()
                if ax_position[0,0] != old_ax_position[0,0] or\
                    ax_position[1,0] != old_ax_position[1,0]:
                    # x extents have moved
                    old_xhist_pos = axHistx.get_position().get_points()
                    # specify new position in terms of [left, bottom, width, height]
                    bottom = old_xhist_pos[0,1]
                    height = old_xhist_pos[1,1] - old_xhist_pos[0,1]
                    left = ax_position[0,0]
                    width = ax_position[1,0] - ax_position[0,0]
                    axHistx.set_position([left, bottom, width, height])
                elif ax_position[0,1] != old_ax_position[0,1] or\
                    ax_position[1,1] != old_ax_position[1,1]:
                    # y extents have moved
                    old_yhist_pos = axHisty.get_position().get_points()
                    # specify new position in terms of [left, bottom, width, height]
                    left = old_yhist_pos[0,0]
                    width = old_yhist_pos[1,0] - old_yhist_pos[0,0]
                    bottom = ax_position[0,1]
                    height = ax_position[1,1] - ax_position[0,1]
                    axHisty.set_position([left, bottom, width, height])


            # add a grassy porous layer background (if porous layer present)
            if self.h_p is not None:
                grass = np.random.rand(80)*self.L[0]
                for g in grass:
                    ax.axvline(x=g, ymax=self.h_p/self.L[1], color='.5')

            # plot any ghost structures
            for plot_func, args in zip(self.plot_structs, 
                                       self.plot_structs_args):
                plot_func(ax, *args)

            # plot ibmesh
            if self.ibmesh is not None:
                line_segments = LineCollection(self.ibmesh)
                line_segments.set_color('k')
                ax.add_collection(line_segments)

            # include tick labels for endpoints
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            if xticks[-1] > self.L[0]:
                xticks[-1] = np.around(self.L[0], 3)
                ax.set_xticks(xticks)
            if yticks[-1] > self.L[1]:
                yticks[-1] = np.around(self.L[1], 3)
                ax.set_yticks(yticks)

            if nohist:
                return ax
            else:
                return ax, axHistx, axHisty


        ########## 3D plot ##########
        else:
            if nohist:
                ax = mplot3d.Axes3D(fig)
            else:
                # chop up axes for a tight layout with histograms
                left_s, width_s = 0.025, 0.45
                bottom_s, height_s = 0.05, 0.9
                bottom_z, bottom_y, bottom_x = 0.05, 0.375, 0.7
                height_h, left_h, width_h = 0.25, 0.575, 0.4

                rect_scatter = [left_s, bottom_s, width_s, height_s]
                rect_histx = [left_h, bottom_x, width_h, height_h]
                rect_histy = [left_h, bottom_y, width_h, height_h]
                rect_histz = [left_h, bottom_z, width_h, height_h]

                # create scatter plot
                ax = mplot3d.Axes3D(fig, rect=rect_scatter) #fig.add_subplot(121, projection='3d')
                ax.set_title('Organism positions')
                # No real solution to 3D aspect ratio...
                #ax.set_aspect('equal','box')

                # histograms
                int_ticks = MaxNLocator(nbins='auto', integer=True)
                axHistx = plt.axes(rect_histx)
                axHistx.set_xlim((0, self.L[0]))
                axHistx.yaxis.set_major_locator(int_ticks)
                axHistx.set_ylabel('X    ', rotation=0)
                axHisty = plt.axes(rect_histy)
                axHisty.set_xlim((0, self.L[1]))
                axHisty.yaxis.set_major_locator(int_ticks)
                axHisty.set_ylabel('Y    ', rotation=0)
                axHistz = plt.axes(rect_histz)
                axHistz.set_xlim((0, self.L[2]))
                axHistz.yaxis.set_major_locator(int_ticks)
                axHistz.set_ylabel('Z    ', rotation=0)

            # 3D plot labels and limits
            ax.set_xlim((0, self.L[0]))
            ax.set_ylim((0, self.L[1]))
            ax.set_zlim((0, self.L[2]))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # add a grassy porous layer background (if porous layer present)
            if self.h_p is not None:
                grass = np.random.rand(120,2)
                grass[:,0] *= self.L[0]
                grass[:,1] *= self.L[1]
                for g in grass:
                    ax.plot([g[0],g[0]], [g[1],g[1]], [0,self.h_p],
                            'k-', alpha=0.5)

            # plot any structures
            for plot_func, args in zip(self.plot_structs, 
                                       self.plot_structs_args):
                plot_func(ax, *args)

            # plot ibmesh
            if self.ibmesh is not None:
                structures = mplot3d.art3d.Poly3DCollection(self.ibmesh)
                structures.set_color('g')
                structures.set_alpha(0.3)
                ax.add_collection3d(structures)

            # include tick labels for endpoints
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            zticks = ax.get_zticks()
            if xticks[-1] > self.L[0]:
                xticks[-1] = np.around(self.L[0], 3)
                ax.set_xticks(xticks)
            if yticks[-1] > self.L[1]:
                yticks[-1] = np.around(self.L[1], 3)
                ax.set_yticks(yticks)
            if zticks[-1] > self.L[2]:
                zticks[-1] = np.around(self.L[2], 3)
                ax.set_zticks(zticks)

            if nohist:
                return ax
            else:
                return ax, axHistx, axHisty, axHistz



    def plot_envir(self, figsize=None):
        '''Plot the environment without the flow, e.g. to verify ibmesh
        formed correctly, dimensions are correct, etc.'''

        if figsize is None:
            if len(self.L) == 2:
                aspectratio = self.L[0]/self.L[1]
                if aspectratio > 1:
                    x_length = np.min((6*aspectratio,12))
                    y_length = 6
                elif aspectratio < 1:
                    x_length = 6
                    y_length = np.min((6/aspectratio,8))
                else:
                    x_length = 6
                    y_length = 6
                # with no histogram plots, can adjust other length in edge cases
                if x_length == 12:
                    y_length = 12/aspectratio
                elif y_length == 8:
                    x_length = 8*aspectratio
                fig = plt.figure(figsize=(x_length,y_length))
            else:
                fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        ax = self._plot_setup(fig, nohist=True)
        plt.show()



    def plot_flow(self, t=None, downsamp=5, interval=500, figsize=None, **kwargs):
        '''Plot the velocity field of the fluid at a given time t or at all
        times if t is None. If t is not in self.flow_times, the nearest time
        will be shown without interpolation.

        For densely sampled velocity fields, specify an int for downsamp to plot
        every nth vector in each direction.

        For time dependent velocity fields, interval is the delay between plotting
        of each time's flow data, in milliseconds. Defaults to 500.
        
        Extra keyword arguments will be passed to pyplot's quiver.

        2D arrow lengths are scaled based on the maximum of all the data over 
        all times.
        '''

        # Locate the flow field that will need plotting, or None if not
        #   time-dependent or we are going to plot all of them.
        if t is not None and self.flow_times is not None:
            loc = np.searchsorted(self.flow_times, t)
            if loc == len(self.flow_times):
                loc = -1
            elif t < self.flow_times[loc]:
                if (self.flow_times[loc]-t) > (t-self.flow_times[loc-1]):
                    loc -= 1
        else:
            loc = None

        if downsamp is None:
            M = 1
        else:
            assert isinstance(downsamp, int), "downsamp must be int or None"
            assert downsamp>0, "downsamp must be a positive int (min 1)"
            M = downsamp

        def animate(n, quiver, kwargs):
            time_text.set_text('time = {:.2f}'.format(self.flow_times[n]))
            if len(self.L) == 2:
                quiver.set_UVC(self.flow[0][n][::M,::M].T,
                               self.flow[1][n][::M,::M].T)
            else:
                quiver = ax.quiver(x,y,z,self.flow[0][n][::M,::M,::M],
                                         self.flow[1][n][::M,::M,::M],
                                         self.flow[2][n][::M,::M,::M], **kwargs)
                fig.canvas.draw()
            return [quiver, time_text]

        if figsize is None:
            if len(self.L) == 2:
                aspectratio = self.L[0]/self.L[1]
                if aspectratio > 1:
                    x_length = np.min((6*aspectratio,12))
                    y_length = 6
                elif aspectratio < 1:
                    x_length = 6
                    y_length = np.min((6/aspectratio,8))
                else:
                    x_length = 6
                    y_length = 6
                # with no histogram plots, can adjust other length in edge cases
                if x_length == 12:
                    y_length = 12/aspectratio
                elif y_length == 8:
                    x_length = 8*aspectratio
                fig = plt.figure(figsize=(x_length,y_length))
            else:
                fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        ax = self._plot_setup(fig, nohist=True)

        ########## 2D Plot #########
        if len(self.L) == 2:
            # get worse case max velocity vector for scaling
            max_u = self.flow[0].max(); max_v = self.flow[1].max()
            max_mag = np.linalg.norm(np.array([max_u,max_v]))
            if len(self.L) == len(self.flow[0].shape) or t is not None:
                # Single-time plot.
                if loc is None:
                    ax.quiver(self.flow_points[0][::M], self.flow_points[1][::M],
                              self.flow[0][::M,::M].T, self.flow[1][::M,::M].T, 
                              scale=None, **kwargs)
                else:
                    ax.quiver(self.flow_points[0][::M], self.flow_points[1][::M],
                              self.flow[0][loc][::M,::M].T,
                              self.flow[1][loc][::M,::M].T, 
                              scale=None, **kwargs)
            else:
                # Animation plot
                # create quiver object
                quiver = ax.quiver(self.flow_points[0][::M], self.flow_points[1][::M], 
                                   self.flow[0][0][::M,::M].T,
                                   self.flow[1][0][::M,::M].T, 
                                   scale=None, **kwargs)
                # textual info
                time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                    fontsize=12)
        ########## 3D Plot #########        
        else:
            x, y, z = np.meshgrid(self.flow_points[0][::M], self.flow_points[1][::M], 
                                  self.flow_points[2][::M], indexing='ij')
            if len(self.L) == len(self.flow[0].shape) or t is not None:
                # Single-time plot.
                if loc is None:
                    ax.quiver(x,y,z,self.flow[0][::M,::M,::M],
                                    self.flow[1][::M,::M,::M],
                                    self.flow[2][::M,::M,::M], **kwargs)
                else:
                    ax.quiver(x,y,z,self.flow[0][loc][::M,::M,::M],
                                    self.flow[1][loc][::M,::M,::M],
                                    self.flow[2][loc][::M,::M,::M], **kwargs)
            else:
                # Animation plot
                quiver = ax.quiver(x,y,z,self.flow[0][0][::M,::M,::M],
                                         self.flow[1][0][::M,::M,::M],
                                         self.flow[2][0][::M,::M,::M], **kwargs)
                # textual info
                time_text = ax.text2D(0.02, 1, 'time = {:.2f}'.format(
                                  self.flow_times[0]),
                                  transform=ax.transAxes, animated=True,
                                  verticalalignment='top', fontsize=12)

        if len(self.L) < len(self.flow[0].shape) and t is None:
            frames = range(len(self.flow_times))
            anim = animation.FuncAnimation(fig, animate, frames=frames,
                                        fargs=(quiver,kwargs),
                                        interval=interval, repeat=False,
                                        blit=True, save_count=len(frames))

        plt.show()



    def plot_2D_vort(self, t=None, clip=None, interval=500, figsize=None):
        '''Plot the vorticity of a 2D fluid at the given time t or at all
        times if t is None. This method will calculate the vorticity on the fly.

        Clip will limit the extents of the color scale.

        For time dependent velocity fields, interval is the delay between plotting
        of each time's flow data, in milliseconds. Defaults to 500.
        '''

        # Locate the flow field that will need plotting, or None if not
        #   time-dependent or we are going to plot all of them.
        assert len(self.L) == 2, "Flow field must be 2D!"

        def animate(n, pc, time_text):
            vort = self.get_2D_vorticity(t_n=n)
            time_text.set_text('time = {:.3f}'.format(self.flow_times[n]))
            pc.set_array(vort.T)
            pc.changed()
            pc.autoscale()
            cbar.update_normal(pc)
            fig.canvas.draw()
            return [pc, time_text]

        if figsize is None:
            aspectratio = self.L[0]/self.L[1]
            if aspectratio > 1:
                x_length = np.min((6*aspectratio,12))
                y_length = 6
            elif aspectratio < 1:
                x_length = 6
                y_length = np.min((6/aspectratio,8))
            else:
                x_length = 6
                y_length = 6
            # with no histogram plots, can adjust other length in edge cases
            if x_length == 12:
                y_length = 12/aspectratio
            elif y_length == 8:
                x_length = 8*aspectratio + 0.75 # leave room for colorbar
            fig = plt.figure(figsize=(x_length,y_length))
        else:
            fig = plt.figure(figsize=figsize)
        ax = self._plot_setup(fig, nohist=True)
        ax.set_aspect('equal')

        if clip is not None:
            norm = colors.Normalize(-abs(clip),abs(clip),clip=True)
        else:
            norm = None

        if len(self.L) == len(self.flow[0].shape):
            # Single-time plot from single-time flow
            vort = self.get_2D_vorticity()
            pc = ax.pcolormesh(self.flow_points[0], self.flow_points[1],
                          vort.T, shading='gouraud', cmap='RdBu', norm=norm)
            axbbox = ax.get_position().get_points()
            cbaxes = fig.add_axes([axbbox[1,0]+0.01, axbbox[0,1], 0.02, axbbox[1,1]-axbbox[0,1]])
            fig.colorbar(pc, cax=cbaxes)
        elif t is not None:
            vort = self.get_2D_vorticity(time=t)
            pc = ax.pcolormesh(self.flow_points[0], self.flow_points[1],
                          vort.T, shading='gouraud', cmap='RdBu', norm=norm)
            ax.text(0.02, 0.95, 'time = {:.3f}'.format(t), transform=ax.transAxes, fontsize=12)
            axbbox = ax.get_position().get_points()
            cbaxes = fig.add_axes([axbbox[1,0]+0.01, axbbox[0,1], 0.02, axbbox[1,1]-axbbox[0,1]])
            fig.colorbar(pc, cax=cbaxes)
        else:
            # Animation plot
            # create pcolormesh object
            # vort = self.get_2D_vorticity(t_n=0)
            pc = ax.pcolormesh([self.flow_points[0]], self.flow_points[1], 
                           np.zeros(self.flow[0].shape[1:]).T, shading='gouraud',
                           cmap='RdBu', norm=norm)
            axbbox = ax.get_position().get_points()
            cbaxes = fig.add_axes([axbbox[1,0]+0.01, axbbox[0,1], 0.02, axbbox[1,1]-axbbox[0,1]])
            cbar = fig.colorbar(pc, cax=cbaxes)
            # textual info
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                fontsize=12)
            frames = range(len(self.flow_times))
            anim = animation.FuncAnimation(fig, animate, frames=frames,
                                        fargs=(pc,time_text),
                                        interval=interval, repeat=False,
                                        blit=False, save_count=len(frames))
        plt.show()



    def plot_2D_FTLE(self, smallest=False, clip_l=None, clip_h=None, figsize=None):
        '''Plot the FTLE field as generated by the calculate_FTLE method. The field 
        will be hard to visualize in 3D, so only 2D is implemented here. For 3D 
        visualization, output the field as a vtk and visualize using VisIt, ParaView, 
        etc.

        TODO: Show a video of 2D slices as a plot of 3D FTLE

        Parameters
        ----------
        smallest : bool, default=False
            If true, plot the negative, smallest, forward-time FTLE as 
            a way of identifying attracting Lagrangian Coherent Structures (see 
            Haller and Sapsis 2011). Otherwise, plot the largest, forward-time 
            FTLE as a way of identifying ridges (separatrix) of LCSs.
        clip_l : float, optional
            lower clip value (below this value, mask points)
        clip_h : float, optional
            upper clip value (above this value, mask points)
        figsize : tuple, optional
            matplotlib figsize
        '''

        if self.FTLE_loc is None:
            print("Error: must generate FTLE field first! Use the calculate_FTLE method of this class.")
            return
        if len(self.L) > 2:
            print("This method is only valid for 2D environments.")
            return

        if figsize is None:
            aspectratio = self.L[0]/self.L[1]
            if aspectratio > 1:
                x_length = np.min((6*aspectratio,12))
                y_length = 6
            elif aspectratio < 1:
                x_length = 6
                y_length = np.min((6/aspectratio,8))
            else:
                x_length = 6
                y_length = 6
            # with no histogram plots, can adjust other length in edge cases
            if x_length == 12:
                y_length = 12/aspectratio
            elif y_length == 8:
                x_length = 8*aspectratio  + 0.75 # leave room for colorbar
            fig = plt.figure(figsize=(x_length,y_length))
        else:
            fig = plt.figure(figsize=figsize)
        ax = self._plot_setup(fig, nohist=True)
        ax.set_aspect('equal')
        if smallest:
            FTLE = -self.FTLE_smallest
        else:
            FTLE = self.FTLE_largest
        if clip_l is not None:
            FTLE = ma.masked_where(FTLE < clip_l, FTLE)
        if clip_h is not None:
            FTLE = ma.masked_where(FTLE > clip_h, FTLE)
        # if clip_l is not None or clip_h is not None:
        #     norm = colors.Normalize(clip_l,clip_h,clip=True)
        # else:
        #     norm = None
        grid_x = np.reshape(self.FTLE_loc[:,0].data, self.FTLE_grid_dim)
        grid_y = np.reshape(self.FTLE_loc[:,1].data, self.FTLE_grid_dim)
        pcm = ax.pcolormesh(grid_x, grid_y, FTLE, shading='gouraud', 
                            cmap='plasma')
        if smallest:
            plt.title('Negative smallest fwrd-time FTLE field, $t_0$={}, $\Delta t$={}.'.format(
                    self.FTLE_t0, self.FTLE_T))
        else:
            plt.title('Largest fwrd-time FTLE field, $t_0$={}, $\Delta t$={}.'.format(
                    self.FTLE_t0, self.FTLE_T))
        axbbox = ax.get_position().get_points()
        cbaxes = fig.add_axes([axbbox[1,0]+0.01, axbbox[0,1], 0.02, axbbox[1,1]-axbbox[0,1]])
        plt.colorbar(pcm, cax=cbaxes)
        plt.show()

