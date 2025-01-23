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
import pandas as pd
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

if dataio.NETCDF:
    from cftime import date2num

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
    x_bndry : {'zero', 'noflux', 'periodic'} as str or [str, str], default='zero'
        agent boundary condition in the x-axis (if the same on both sides), or 
        [left bndry condition, right bndry condition]. Choices are 'zero', 
        'noflux', and 'periodic'. Agents leaving a zero boundary condition will 
        be marked as masked and cease to be updated or plotted afterward. In the 
        noflux case, agents will undergo a sliding collision with the boundary. 
        Movement that would have occurred through the boundary will be projected 
        onto the boundary instead. In the periodic case, agents leaving one side 
        of the domain will reenter on the other side.
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
    ibmesh_color : matplotlib color format, optional
        color of the ibmesh. Defaults to black in 2D, 'dimgrey' in 3D.

    Attributes
    ----------
    L : list of floats
        length of the domain [x, y, [z]] in the stated units
    units : str
        A place to remind yourself of the spatial units you are working in. Note 
        that none of the methods of this class use this attribute in any way, 
        so it's probably best to work in meters.
    bndry :  list of lists, each with two of {'zero', 'noflux', 'periodic'}
        Boundary conditions in each direction [x, y, [z]] for agents
    swarms : list of swarm objects
        The swarms that belong to this environment
    time : float
        current environment time
    time_history : list of floats
        list of past time states
    flow : list of ndarrays or fCubicSpline objects
        [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k]),
            z-vel field ndarray (if 3D)]. i is x index, j is y index, with the 
            value of x and y increasing as the index increases. Arrays get 
            replaced by fCubicSpline objects (if the fluid velocity is 
            temporally varying) when they are first needed.
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
    netcdf : netCDF4 Dataset object
        only present if a netCDF file has been loaded for reading
    ibmesh : ndarray
        This is either an Nx2x2 array (2D, line mesh) or an Nx3x3 array (3D, 
        triangular mesh) specifying internal mesh structures that agents will 
        treat as solid barriers. Each value of the first index into the array 
        specifies a mesh element, and each 1x2 or 1x3 contained in that index 
        is a point in space (so, two for a line, three for a triangle). Avoid 
        mesh structures that intersect with a periodic boundary - behavior 
        related to this is not implemented.
    max_meshpt_dist : float
        maximum length of a mesh segment in ibmesh, typically determined 
        automatically. This is used in search algorithms to winnow down the 
        number of mesh elements to search for intersections of movement.
    ibmesh_color : matplotlib color format
        color of the ibmesh. Defaults to black in 2D, 'dimgrey' in 3D.
    plot_structs : list of function handles
        List of functions that plot additional environment structures which the 
        agents are unaware of. E.g., for visual purposes only.
    plot_structs_args : List of tuples
        List of argument tuples to be passed to plot_structs functions, after 
        ax (the plot axis object) is passed
    FTLE_largest : ndarray
        FTLE field calculated using the largest eigenvalue
    FTLE_smallest : ndarray
        FTLE field calculated using the smallest eigenvalue. take the negative 
        of this to get backward-time information
    FTLE_loc : Nx2 masked ndarray
        spatial points on which the FTLE mesh was calculated
    FTLE_loc_end : Nx2 masked ndarray
        final locations for each of the FTLE mesh points
    FTLE_t0 : float
        start-time for the FTLE calculation
    FTLE_T : float
        integration time for FTLE
    FTLE_grid_dim : tuple of int
        grid dimension for the FTLE calculation (x,y,[z])
    mag_grad : ndarray
        acts as a cache for the gradient of magnitude of the fluid velocity
    mag_grad_time : float
        simulation time at which magnitude gradient above was calculated
    DuDt : list of ndarrays
        material derivative cache
    DuDt_time : float
        simulation time at which material derivative was calculated
    dt_interp : list of PPoly objects
        Used for temporal derivative interpolation. Set by dudt method.

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
                 U=None, init_swarms=None, units='m', ibmesh_color=None):

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
        self.mag_grad = None
        self.mag_grad_time = None
        self.DuDt = None
        self.DuDt_time = None
        self.flow = flow
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
        if ibmesh_color is None:
            if Lz is None:
                self.ibmesh_color = 'k'
            else:
                self.ibmesh_color = 'dimgrey'
        else:
            self.ibmesh_color = ibmesh_color

        ##### Environment Structure Plotting #####

        # NOTE: the agents do not interact with these structures; for plotting only!

        # List of functions that plot additional environment structures
        self.plot_structs = []
        # List of arguments tuples to be passed to these functions, after ax
        self.plot_structs_args = []

        ##### Initalize Time #####

        # By default, time is updated whenever an individual swarm moves (swarm.move()),
        #   or when all swarms in the environment are collectively moved.
        self.time = 0.0
        self.time_history = []

        ##### FTLE fields #####
        self.FTLE_largest = None
        self.FTLE_smallest = None # take negative to get backward-time picture
        self.FTLE_loc = None
        self.FTLE_loc_end = None
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
        auto_regrid : bool, default=True
            IB2d always has periodic BC and returns a VTK with fluid specified 
            at the center of the mesh cells. Regrid based on this information to
            make the fluid periodic within Planktos and to fill out the domain.
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

        ### Convert environment dimensions and add back the periodic gridpoints ###
        self.L = [self.flow_points[dim][-1] for dim in range(2)]
        self.wrap_flow(periodic_dim=(True, True))



    def read_IBAMR3d_vtk_data(self, filename, vel_conv=None, grid_conv=None):
        '''Reads in vtk flow data from a single source and sets environment
        variables accordingly. The resulting flow will be time invarient. It is
        assumed this data is a rectilinear grid.

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
            raise FileNotFoundError("File {} not found!".format(filename))

        data, mesh, time = dataio.read_vtk_Rectilinear_Grid_Vector(filename)

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



    def load_NetCDF(self, filename):
        '''Load a NetCDF file into the netcdf attribute of the environment. 
        Does not automatically read in any data.

        Because NetCDF files can contain multiple data sets with different 
        dimension names and associated metadata, and because it may be desirable 
        to explore the data set first and/or load only a subset of the data, 
        this method just loads the Dataset into the environment object.
        See the documentation/tutorial for netCDF4 on ways to read the metadata
        for the loaded Dataset. See read_NetCDF_flow for reading in data from a 
        loaded NetCDF dataset.

        Parameters
        ----------
        filename : string
            path and filename of the NetCDF file, including extension

        See Also
        --------
        read_NetCDF_flow : read in data from a loaded NetCDF dataset
        '''
        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(str(filename)))

        self.netcdf = dataio.load_netcdf(filename)



    def read_NetCDF_flow(self, flow_x, flow_y=None, flow_z=None, vec_idx=None, 
                         dim_reorder=None, x_name=None, y_name=None, z_name=None, 
                         time_name=None, conv_time=False):
        '''Read NetCDF fluid data into the environment. Must first have loaded a 
        NetCDF dataset with load_NetCDF.

        The default expectation is that the x, y, and z components of the fluid 
        velocity data are specified in separate Variables. If that is not the 
        case, then the index of the Variable dimension which specifies the 
        component of the vector must be supplied in the parameter vec_idx 
        (base 0). It will be assumed that the ordering in this Dimension is x, 
        y, then z components.

        This method assumes that mesh point values in the x, y, and z directions 
        are given by Variables which have the same name as the Dimensions of the 
        fluid flow Variables. If so, the method will automatically find and load 
        them. Otherwise, strings must be specified for x_name, y_name, and 
        z_name giving the Variable names. The same goes for the time mesh data.

        **Blurb about time conversion**

        If the coordinate variables have a Unit attribute, this will be loaded 
        as the environment's units.

        Parameters
        ----------
        flow_x : string
            Variable name (or path, if the variable is inside group) to the 
            fluid velocity data for the x-direction, or for all directions.
        flow_y : string, optional
            Variable name (or path, if the variable is inside group) to the 
            fluid velocity data for the y-direction. If all directions are in 
            the same variable, this does not need to be supplied, but vec_idx 
            must be supplied instead.
        flow_z : string, optional
            Similar to flow_y.
        vec_idx : int, optional
            Index of the variable dimension along which the different components 
            of the velocity are given. Leave as None if the components are given 
            in separate variables.
        dim_reorder : tuple or list of ints, optional
            Fluid velocity data is stored within this class in the dimensional 
            ordering ([time], x, y, [z]). If this matches the NetCDF ordering, 
            then no adjustment is necessary. Otherwise, this must be specified 
            as a tuple or list which contains a permutation of [0,1,..,N-1] 
            where N is the number of spatial-temporal dimensions and the numbers 
            correspond to where each dimension of the NetCDF variable should go 
            in the Planktos ordering. If the NetCDF variable has a dimension 
            specifying the component of the fluid velocity, ignore it and do not 
            include it in the permutation list.
        x_name : string, optional
            Variable name (or path, if the variable is inside a group) for the 
            coordinate variable corresponding with the x spatial direction. Only 
            necessary if different from the NetCDF Dimension name.
        y_name : string, optional
            Similar to x_name.
        z_name : string, optional
            Similar to x_name.
        time_name : string, optional
            Variable name (or path, if the variable is inside a group) for the 
            coordinate variable corresponding with the time dimension. Only 
            necessary if different from the NetCDF Dimension name.
        conv_time : bool, default=False
            If the time variable is not numerical, set this to True in order to 
            convert from a time format relative to a fixed date using a certain 
            calendar to floating point numerical values. In this case, it will 
            be expected that the time variable is a sequence of datetime 
            objects, so conversion of this variable from a string to the python 
            datetime object may be required first.
            Uses the date2num function provided by cftime. See Notes below for 
            expected formatting in this case.

        See Also
        --------
        load_NetCDF : Load a NetCDF file into the netcdf environment attribute.

        Notes
        -----
        In order for the date2num conversion to work, units and a calendar must 
        be specified in the coordinate variable for time. Specifically, the 
        attribute of the time variable named "units" must be a string of the 
        form "<time units> since <reference time>". <time units> can be days, 
        hours, minutes, seconds, milliseconds or microseconds. <reference time> 
        is the time origin. months since is allowed only for the 360_day calendar 
        and common_years since is allowed only for the 365_day calendar. The 
        attribute of the time variable named "calendar" must be a string 
        descrbing the calendar to be used in the time calculations. All values 
        in the CF metadata convention are supported. Valid calendars ‘standard’, 
        ‘gregorian’, ‘proleptic_gregorian’ ‘noleap’, ‘365_day’, ‘360_day’, 
        ‘julian’, ‘all_leap’, ‘366_day’. Default is None which means the 
        calendar associated with the first input datetime instance will be used.
        '''
        
        if flow_y is None:
            assert vec_idx is not None, "vec_idx must be specified if there is only one fluid variable"

        # Get the flow data, permuting the axes as necessary
        flow = []
        if vec_idx is None:
            if dim_reorder is None:
                flow.append(self.netcdf[flow_x][:])
                flow.append(self.netcdf[flow_y][:])
                if flow_z is not None:
                    flow.append(self.netcdf[flow_z][:])
            else:
                flow.append(np.transpose(self.netcdf[flow_x][:], dim_reorder))
                flow.append(np.transpose(self.netcdf[flow_y][:], dim_reorder))
                if flow_z is not None:
                    flow.append(np.transpose(self.netcdf[flow_z][:], dim_reorder))
        else:
            dim = self.netcdf[flow_x].shape[vec_idx]
            full_flow = np.moveaxis(self.netcdf[flow_x][:], vec_idx, 0)
            if dim_reorder is None:
                for d in range(dim):
                    flow.append(full_flow[d,...])
            else:
                for d in range(dim):
                    flow.append(np.transpose(full_flow[d,...], dim_reorder))

        # Detect time dependence and spatial dimension
        ts_dim = len(flow[0].shape)
        if vec_idx is None:
            if flow_z is None:
                time_dep = ts_dim == 3
            else:
                time_dep = ts_dim == 4
        else:
            time_dep = dim != ts_dim
        if time_dep:
            s_dim = ts_dim-1
        else:
            s_dim = ts_dim

        # Get dimension ordering inside NetCDF
        if vec_idx is None and dim_reorder is None:
            dim_order = np.arange(ts_dim)
        elif dim_reorder is None:
            dim_order = np.arange(ts_dim)
            dim_order[vec_idx:] += 1
        else:
            if vec_idx is None:
                dim_idx_reorder = dim_reorder
            else:
                dim_idx_reorder = list(dim_reorder).insert(vec_idx, 99)
            dim_order = []
            for d in range(len(dim_reorder)):
                for idx, val in enumerate(dim_idx_reorder):
                    if d == val:
                        dim_order.append(idx)
                        break

        # Get spatial mesh data
        if time_dep:
            t = 1
        else:
            t = 0
        if x_name is None:
            x_name = self.netcdf[flow_x].dimensions[dim_order[0+t]]
        flow_points_x = self.netcdf[x_name][:]
        try:
            s_units = self.netcdf[x_name].units
        except AttributeError:
            s_units = None
        if y_name is None:
            y_name = self.netcdf[flow_x].dimensions[dim_order[1+t]]
        flow_points_y = self.netcdf[y_name][:]
        if s_dim > 2:
            if z_name is None:
                z_name = self.netcdf[flow_x].dimensions[dim_order[2+t]]
            flow_points_z = self.netcdf[z_name][:]

        # Get time data, converting if necessary
        if time_dep:
            if time_name is None:
                time_name = self.netcdf[flow_x].dimensions[dim_order[0]]
            if conv_time:
                flow_times = date2num(self.netcdf[time_name][:], 
                                      units=self.netcdf[time_name].units, 
                                      calendar=self.netcdf[time_name].calendar)
            else:
                flow_times = self.netcdf[time_name][:]
        
        # Set environment variables and reset LLC to origin
        self.flow = flow
        if s_units is not None:
            self.units = s_units
        if s_dim == 2:
            self.flow_points = (flow_points_x-flow_points_x[0], 
                                flow_points_y-flow_points_y[0])
        else:
            self.flow_points = (flow_points_x-flow_points_x[0], 
                                flow_points_y-flow_points_y[0],
                                flow_points_z-flow_points_z[0])
        self.L = [self.flow_points[dim][-1] for dim in range(s_dim)]
        if time_dep:
            self.flow_times = flow_times
        self.__reset_flow_variables()
        if s_dim == 2:
            self.fluid_domain_LLC = (flow_points_x[0], flow_points_y[0])
        else:
            self.fluid_domain_LLC = (flow_points_x[0], flow_points_y[0], flow_points_z[0])



    def wrap_flow(self, periodic_dim=(True, True, False)):
        '''In some cases, software may print out fluid velocity data that omits 
        the velocities at the right boundaries in spatial dimensions that are 
        meant to be periodic. This helper function restores that data by copying 
        everything over. 3rd dimension will automatically be ignored if 2D.

        Parameters
        ----------
        periodic_dim : list of 3 bool, default=[True, True, False]
            True if that spatial dimension is periodic, otherwise False
        '''

        dim = len(self.flow_points)
        if dim == len(self.flow[0].shape):
            TIME_DEP = False
        else:
            TIME_DEP = True
                
        dx = np.array([self.flow_points[d][-1]-self.flow_points[d][-2] 
                       for d in range(dim)])
        
        # find new flow field shape
        new_flow_shape = np.array(self.flow[0].shape)
        if not TIME_DEP:
            new_flow_shape += 1*np.array(periodic_dim)
        else:
            new_flow_shape[1:] += 1*np.array(periodic_dim)

        # create new flow field, putting old data in lower left corner
        new_flow = [np.zeros(new_flow_shape) for d in range(dim)]
        if TIME_DEP:
            old_shape = self.flow[0].shape[1:]
        else:
            old_shape = self.flow[0].shape
        for d in range(dim):
            if dim == 2:
                new_flow[d][...,:old_shape[0],:old_shape[1]] = self.flow[d]
            else:
                new_flow[d][...,:old_shape[0],:old_shape[1],:old_shape[2]] = self.flow[d]
        # replace old flow field
        self.flow = new_flow

        # fill in the new edges and update flow points
        flow_points = []
        for d in range(dim):
            if periodic_dim[d]:
                flow_points.append(np.append(self.flow_points[d], 
                                   self.flow_points[d][-1]+dx[d]))
                for dd in range(dim):
                    if d == 0 and not TIME_DEP:
                        self.flow[dd][-1,...] = self.flow[dd][0,...]
                    elif d == 0 and TIME_DEP:
                        self.flow[dd][:,-1,...] = self.flow[dd][:,0,...]
                    elif d == 1 and not TIME_DEP:
                        self.flow[dd][:,-1,...] = self.flow[dd][:,0,...]
                    elif d == 1 and TIME_DEP:
                        self.flow[dd][:,:,-1,...] = self.flow[dd][:,:,0,...]
                    else:
                        self.flow[dd][...,-1] = self.flow[dd][...,0]
            else:
                flow_points.append(self.flow_points[d])
        
        # replace old flow points
        self.flow_points = tuple(flow_points)

        # replace domain length
        self.L = [self.flow_points[d][-1] for d in range(dim)]
        self.__reset_flow_variables()



    def center_cell_regrid(self, periodic_dim=(False, False, False)):
        '''Re-grids data that was specified at the center of cells instead of
        at the corners.

        NOTE! This needs to be called *before* any immersed meshes are loaded.
        It will NOT look for and properly shift these meshes.
        
        Software has a tendency to output data files where the fluid mesh is 
        specified at the center of cells rather than at the corners. This will 
        be readily apparent if Planktos loads your fluid velocity data and 
        reports spacial dimensions one dx, dy, and dz smaller than you were 
        expecting. To fix this, Planktos will interpolate/extrapolate the fluid 
        velocity mesh using the default method to get additional grid points on 
        the edge of the domain.

        Periodicity can be enforced in specified dimensions.

        Parameters
        ----------
        periodic_dim : list-like of 2 or 3 bool, default=(True, True, False)
            True if that spatial dimension is periodic, otherwise False.
            The 3rd entry will be ignored in the 2D case.
        '''
        
        # Detect cell width in each dimension based on the first two coordinates 
        #   in each spatial dimension
        dx = self.flow_points[0][1] - self.flow_points[0][0]
        dy = self.flow_points[1][1] - self.flow_points[1][0]
        if len(self.L) > 2:
            dz = self.flow_points[2][1] - self.flow_points[2][0]
            DIM3 = True
        else:
            DIM3 = False

        ### Create a list of positions at which we need to extrapolate the ###
        ###   velocity field                                               ###
        x_ends = [-dx/2, self.flow_points[0][-1]+dx/2]
        y_ends = [-dy/2, self.flow_points[1][-1]+dy/2]
        bndry_list = []
        if not DIM3:
            # edges
            bndry_list += [[x, y_ends[0]] for x in self.flow_points[0]]
            bndry_list += [[x, y_ends[1]] for x in self.flow_points[0]]
            bndry_list += [[x_ends[0], y] for y in self.flow_points[1]]
            bndry_list += [[x_ends[1], y] for y in self.flow_points[1]]
            # points
            bndry_list += [[x_ends[0],y_ends[0]],[x_ends[0],y_ends[1]],
                           [x_ends[1],y_ends[0]],[x_ends[1],y_ends[1]]]
        else:
            z_ends = [-dz/2, self.flow_points[2][-1]+dz/2]
            # sides
            bndry_list += [[x,y,z_ends[0]] for x in self.flow_points[0] for y in self.flow_points[1]]
            bndry_list += [[x,y,z_ends[1]] for x in self.flow_points[0] for y in self.flow_points[1]]
            bndry_list += [[x,y_ends[0],z] for x in self.flow_points[0] for z in self.flow_points[2]]
            bndry_list += [[x,y_ends[1],z] for x in self.flow_points[0] for z in self.flow_points[2]]
            bndry_list += [[x_ends[0],y,z] for y in self.flow_points[1] for z in self.flow_points[2]]
            bndry_list += [[x_ends[1],y,z] for y in self.flow_points[1] for z in self.flow_points[2]]
            # edges
            bndry_list += [[x, y_ends[0], z_ends[0]] for x in self.flow_points[0]]
            bndry_list += [[x, y_ends[0], z_ends[1]] for x in self.flow_points[0]]
            bndry_list += [[x, y_ends[1], z_ends[0]] for x in self.flow_points[0]]
            bndry_list += [[x, y_ends[1], z_ends[1]] for x in self.flow_points[0]]
            bndry_list += [[x_ends[0], y, z_ends[0]] for y in self.flow_points[1]]
            bndry_list += [[x_ends[0], y, z_ends[1]] for y in self.flow_points[1]]
            bndry_list += [[x_ends[1], y, z_ends[0]] for y in self.flow_points[1]]
            bndry_list += [[x_ends[1], y, z_ends[1]] for y in self.flow_points[1]]
            bndry_list += [[x_ends[0], y_ends[0], z] for z in self.flow_points[2]]
            bndry_list += [[x_ends[0], y_ends[1], z] for z in self.flow_points[2]]
            bndry_list += [[x_ends[1], y_ends[0], z] for z in self.flow_points[2]]
            bndry_list += [[x_ends[1], y_ends[1], z] for z in self.flow_points[2]]
            # points
            bndry_list += [[x_ends[0],y_ends[0],z_ends[0]],
                           [x_ends[0],y_ends[0],z_ends[1]],
                           [x_ends[0],y_ends[1],z_ends[0]],
                           [x_ends[1],y_ends[0],z_ends[1]],
                           [x_ends[0],y_ends[1],z_ends[1]],
                           [x_ends[1],y_ends[0],z_ends[1]],
                           [x_ends[1],y_ends[1],z_ends[0]],
                           [x_ends[1],y_ends[1],z_ends[1]]]

        ### Include periodicity, if applicable, by extending out the fluid field ###
        flowshape = np.array(self.flow[0].shape)
        idx = []
        if len(self.flow[0].shape) == len(self.L):
            # non time-varying flow
            startdim = 0
        else:
            startdim = 1
        for ii in range(2):
            if periodic_dim[ii]:
                flowshape[startdim+ii] += 2
                idx.append([1,flowshape[startdim+ii]-1])
            else:
                idx.append([0,flowshape[startdim+ii]])
        if DIM3:
            if periodic_dim[2]:
                flowshape[startdim+2] += 2
                idx.append([1,flowshape[startdim+2]-1])
            else:
                idx.append([0,flowshape[startdim+2]])
        
        if DIM3:
            flow = [np.zeros(flowshape) for ii in range(3)]
            flow_points = []
            for ii in range(3):
                flow[ii][...,idx[0][0]:idx[0][1],idx[1][0]:idx[1][1],idx[2][0]:idx[2][1]] = self.flow[ii]
            if periodic_dim[0]:
                for ii in range(3):
                    flow[ii][...,0,:,:] = flow[ii][...,-2,:,:]
                    flow[ii][...,-1,:,:] = flow[ii][...,1,:,:]
                flow_points.append(np.insert(self.flow_points[0], # what
                                             [0,len(self.flow_points[0])], # loc
                                             [-dx,self.flow_points[0][-1]+dx])) # vals
            else:
                flow_points.append(self.flow_points[0])
            if periodic_dim[1]:
                for ii in range(3):
                    flow[ii][...,0,:] = flow[ii][...,-2,:]
                    flow[ii][...,-1,:] = flow[ii][...,1,:]
                flow_points.append(np.insert(self.flow_points[1],
                                             [0,len(self.flow_points[1])],
                                             [-dy,self.flow_points[1][-1]+dy]))
            else:
                flow_points.append(self.flow_points[1])
            if periodic_dim[2]:
                for ii in range(3):
                    flow[ii][...,0] = flow[ii][...,-2]
                    flow[ii][...,-1] = flow[ii][...,1]
                flow_points.append(np.insert(self.flow_points[2],
                                             [0,len(self.flow_points[2])],
                                             [-dz,self.flow_points[2][-1]+dz]))
            else:
                flow_points.append(self.flow_points[2])
        else:
            flow = [np.zeros(flowshape) for ii in range(2)]
            flow_points = []
            for ii in range(2):
                flow[ii][...,idx[0][0]:idx[0][1],idx[1][0]:idx[1][1]] = self.flow[ii]
            if periodic_dim[0]:
                for ii in range(2):
                    flow[ii][...,0,:] = flow[ii][...,-2,:]
                    flow[ii][...,-1,:] = flow[ii][...,1,:]
                flow_points.append(np.insert(self.flow_points[0], # what
                                             [0,len(self.flow_points[0])], # loc
                                             [-dx,self.flow_points[0][-1]+dx])) # vals
            else:
                flow_points.append(self.flow_points[0])
            if periodic_dim[1]:
                for ii in range(2):
                    flow[ii][...,0] = flow[ii][...,-2]
                    flow[ii][...,-1] = flow[ii][...,1]
                flow_points.append(np.insert(self.flow_points[1],
                                             [0,len(self.flow_points[1])],
                                             [-dy,self.flow_points[1][-1]+dy]))
            else:
                flow_points.append(self.flow_points[1])

        ### Interpolate the new points ###
        if startdim == 0:
            # non time-varying flow
            new_vecs = self.interpolate_flow(bndry_list, flow, flow_points)
        else:
            new_vecs = []
            for t_idx in range(self.flow[0].shape[0]):
                this_flow = [flow[ii][t_idx,...] for ii in range(len(flow))]
                new_vecs.append(self.interpolate_flow(bndry_list, this_flow, flow_points))

        ### Incorporate the new points into the fluid field and mesh ###
        if DIM3:
            intervals = [dx,dy,dz]
        else:
            intervals = [dx,dy]
        flow_points = [np.insert(self.flow_points[ii]+interval/2,
                                 [0,len(self.flow_points[ii])],
                                 [0,self.flow_points[ii][-1]+interval])
                                 for ii,interval in enumerate(intervals)]
        flowshape = np.array(self.flow[0].shape)
        if startdim == 0:
            flowshape += 2
        else:
            flowshape[1:] += 2
        shp = [len(points) for points in self.flow_points]

        def bndry_add3d(fshape, shp, this_vecs):
            f = [np.zeros(fshape) for ii in range(3)]
            for dim in range(3):
                # sides
                f[dim][1:-1,1:-1,0] = np.reshape(this_vecs[:shp[0]*shp[1],dim],(shp[0],shp[1]))
                s = shp[0]*shp[1]
                f[dim][1:-1,1:-1,-1] = np.reshape(this_vecs[s:s+shp[0]*shp[1],dim],(shp[0],shp[1]))
                s += shp[0]*shp[1]
                f[dim][1:-1,0,1:-1] = np.reshape(this_vecs[s:s+shp[0]*shp[2],dim],(shp[0],shp[2]))
                s += shp[0]*shp[2]
                f[dim][1:-1,-1,1:-1] = np.reshape(this_vecs[s:s+shp[0]*shp[2],dim],(shp[0],shp[2]))
                s += shp[0]*shp[2]
                f[dim][0,1:-1,1:-1] = np.reshape(this_vecs[s:s+shp[1]*shp[2],dim],(shp[1],shp[2]))
                s += shp[1]*shp[2]
                f[dim][-1,1:-1,1:-1] = np.reshape(this_vecs[s:s+shp[1]*shp[2],dim],(shp[1],shp[2]))
                s += shp[1]*shp[2]
                # edges
                f[dim][1:-1,0,0] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
                f[dim][1:-1,0,-1] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
                f[dim][1:-1,-1,0] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
                f[dim][1:-1,-1,-1] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
                f[dim][0,1:-1,0] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
                f[dim][0,1:-1,-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
                f[dim][-1,1:-1,0] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
                f[dim][-1,1:-1,-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
                f[dim][0,0,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
                f[dim][0,-1,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
                f[dim][-1,0,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
                f[dim][-1,-1,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
                # points
                f[dim][0,0,0] = this_vecs[s,dim]
                f[dim][0,0,-1] = this_vecs[s+1,dim]
                f[dim][0,-1,0] = this_vecs[s+2,dim]
                f[dim][-1,0,0] = this_vecs[s+3,dim]
                f[dim][0,-1,-1] = this_vecs[s+4,dim]
                f[dim][-1,0,-1] = this_vecs[s+5,dim]
                f[dim][-1,-1,0] = this_vecs[s+6,dim]
                f[dim][-1,-1,-1] = this_vecs[s+7,dim]
            return f

        def bndry_add2d(fshape, shp, this_vecs):
            f = [np.zeros(fshape) for ii in range(2)]
            for dim in range(2):
                s=0
                # edges
                f[dim][1:-1,0] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
                f[dim][1:-1,-1] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
                f[dim][0,1:-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
                f[dim][-1,1:-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
                # points
                f[dim][0,0] = this_vecs[s,dim]
                f[dim][0,-1] = this_vecs[s+1,dim]
                f[dim][-1,0] = this_vecs[s+2,dim]
                f[dim][-1,-1] = this_vecs[s+3,dim]
            return f

        if DIM3 and startdim == 0:
            # time invariant, 3D
            flow = bndry_add3d(flowshape, shp, new_vecs)
        elif DIM3 and startdim == 1:
            flow = [np.zeros(flowshape) for ii in range(3)]
            for n, this_vecs in enumerate(new_vecs):
                f = bndry_add3d(flowshape[1:], shp, this_vecs)
                flow[0][n,...]=f[0]; flow[1][n,...]=f[1]; flow[2][n,...]=f[2]
        elif not DIM3 and startdim == 0:
            flow = bndry_add2d(flowshape, shp, new_vecs)
        else:
            flow = [np.zeros(flowshape) for ii in range(2)]
            for n, this_vecs in enumerate(new_vecs):
                f = bndry_add2d(flowshape[1:], shp, this_vecs)
                flow[0][n,...]=f[0]; flow[1][n,...]=f[1]

        ### Add back the original fluid data ###
        for dim in range(len(flow)):
            if DIM3:
                flow[dim][...,1:-1,1:-1,1:-1] = self.flow[dim]
            else:
                flow[dim][...,1:-1,1:-1] = self.flow[dim]

        ### Replace fluid and update domain ###
        self.flow_points = tuple(flow_points)
        self.flow = flow
        self.L = [self.flow_points[d][-1] for d in range(len(flow_points))]
        self.fluid_domain_LLC = tuple(np.array(self.fluid_domain_LLC)-np.array(intervals)*0.5)
        self.__reset_flow_variables()



    def read_stl_mesh_data(self, filename, unit_conv=None):
        '''Reads in 3D mesh data from an ascii or binary stl file. Must have
        the numpy-stl library installed. It is assumed that the coordinate
        system of the stl mesh matches the coordinate system of the flow field.
        Thus, the mesh will be translated using the flow LLC if necessary.
        
        Avoid mesh structures that intersect with a periodic boundary; behavior 
        related to this is not implemented.

        Parameters
        ----------
        filename : string
            filename of data to read, incl. file extension
        unit_conv : float, optional
            scalar to multiply the mesh by in order to convert units
        '''

        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))

        ibmesh, self.max_meshpt_dist = dataio.read_stl_mesh(filename, unit_conv)

        # shift coordinates to match any shift that happened in flow data
        if self.fluid_domain_LLC is not None:
            for ii in range(3):
                ibmesh[:,:,ii] -= self.fluid_domain_LLC[ii]
        
        self.ibmesh = ibmesh



    def read_IB2d_vertex_data(self, filename, res_factor=0.501, res=None):
        '''Reads in 2D vertex data from a .vertex file (IB2d). Assumes that any 
        vertices closer than res_factor (default is half + a bit for numerical 
        stability) times the Eulerian mesh resolution are connected linearly. 
        This method gets the Eulerian mesh resolution from the fluid velocity 
        data, so the flow data must be imported first. 
        
        Alternatively, you can pass in the Eulerian mesh resolution directly to 
        the res parameter to get the mesh in absence of any fluid velocity 
        field. Calculate it from the IB2d input file by taking the domain length 
        and dividing by the number of Eulerian grid points: Lx/Nx and Ly/Ny. The 
        smaller of the two numbers should be used.

        Avoid mesh structures that intersect with a periodic boundary; behavior 
        related to this is not implemented.

        Parameters
        ----------
        filename : string
            vertex file to load
        res_factor : float, default=0.501
            this times the Eulerian mesh resolution is the radius that will be
            used.
        res : float, optional
            pass the Eulerian mesh resolution in directly, instead of 
            calculating it from a loaded fluid velocity field
        '''

        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))
        if res is None:
            assert self.flow_points is not None, "Must import flow data first!"
            dists = np.concatenate([self.flow_points[ii][2:-1]-self.flow_points[ii][1:-2]
                                    for ii in range(2)])
            Eulerian_res = dists.min()
        else:
            Eulerian_res = res

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
        '''Tile fluid flow and immersed meshes a number of times in the x and/or 
        y directions. While obviously this works best if the fluid is periodic 
        in the direction(s) being tiled, this will not be enforced. Instead, it 
        will just be assumed that the domain edges are equivalent, and only the
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
        print("Fluid tiled. Planktos domain size is now {}.".format(self.L))
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

        res_x = np.max(self.flow_points[0][1:]-self.flow_points[0][:-1])
        res_y = np.max(self.flow_points[1][1:]-self.flow_points[1][:-1])
        if not TIME_DEP:
            for dim in range(len(self.L)):
                # first, extend in x direction
                if x_minus+x_plus > 0:
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][0,...]]) for ii in range(x_minus)
                        ) + (self.flow[dim][:],) + tuple(
                        np.array([self.flow[dim][-1,...]]) for jj in range(x_plus)
                        ), axis=0)
                # next, extend in y direction. This requires a shuffling
                #   (tranpose) of dimensions...
                if DIM3 and y_minus+y_plus > 0:
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,0,:]]).transpose(1,0,2)
                        for ii in range(y_minus)
                        ) + (self.flow[dim][:],) + tuple(
                        np.array([self.flow[dim][:,-1,:]]).transpose(1,0,2)
                        for jj in range(y_plus)
                        ), axis=1)
                elif y_minus+y_plus > 0:
                    self.flow[dim] = np.concatenate(tuple(
                        np.array([self.flow[dim][:,0]]).T
                        for ii in range(y_minus)
                        ) + (self.flow[dim][:],) + tuple(
                        np.array([self.flow[dim][:,-1]]).T
                        for jj in range(y_plus)
                        ), axis=1)

        else:
            for dim in range(len(self.L)):
                if DIM3:
                    # first, extend in x direction
                    if x_minus+x_plus > 0:
                        self.flow[dim] = np.concatenate(tuple(
                            np.array([self.flow[dim][:,0,...]]).transpose(1,0,2,3) 
                            for ii in range(x_minus)
                            ) + (self.flow[dim][:],) + tuple(
                            np.array([self.flow[dim][:,-1,...]]).transpose(1,0,2,3) 
                            for jj in range(x_plus)
                            ), axis=1)
                    # next, extend in y direction
                    if y_minus+y_plus > 0:
                        self.flow[dim] = np.concatenate(tuple(
                            np.array([self.flow[dim][:,:,0,:]]).transpose(1,2,0,3)
                            for ii in range(y_minus)
                            ) + (self.flow[dim][:],) + tuple(
                            np.array([self.flow[dim][:,:,-1,:]]).transpose(1,2,0,3)
                            for jj in range(y_plus)
                            ), axis=2)
                else:
                    # first, extend in x direction
                    if x_minus+x_plus > 0:
                        self.flow[dim] = np.concatenate(tuple(
                            np.array([self.flow[dim][:,0,:]]).transpose(1,0,2) 
                            for ii in range(x_minus)
                            ) + (self.flow[dim][:],) + tuple(
                            np.array([self.flow[dim][:,-1,:]]).transpose(1,0,2) 
                            for jj in range(x_plus)
                            ), axis=1)
                    # next, extend in y direction
                    if y_minus+y_plus > 0:
                        self.flow[dim] = np.concatenate(tuple(
                            np.array([self.flow[dim][:,:,0]]).tranpsose(1,2,0)
                            for ii in range(y_minus)
                            ) + (self.flow[dim][:],) + tuple(
                            np.array([self.flow[dim][:,:,-1]]).transpose(1,2,0)
                            for jj in range(y_plus)
                            ), axis=2)

        # update environment dimensions and meshes
        new_points = []
        self.orig_L = tuple(self.L[:2])
        self.L[0] += res_x*(x_minus+x_plus)
        self.L[1] += res_y*(y_minus+y_plus)
        if x_minus+x_plus > 0:
            new_points.append(np.concatenate(
                [self.flow_points[0][0]-res_x*np.arange(x_minus,0,-1)]+
                [self.flow_points[0]]+
                [self.flow_points[0][-1]+res_x*np.arange(1,x_plus+1)]))
        else:
            new_points.append(self.flow_points[0])
        if y_minus+y_plus > 0:
            new_points.append(np.concatenate(
                [self.flow_points[1][0]-res_y*np.arange(y_minus,0,-1)]+
                [self.flow_points[1]]+
                [self.flow_points[1][-1]+res_y*np.arange(1,y_plus+1)]))
        else:
            new_points.append(self.flow_points[1])

        if DIM3:
            new_points.append(self.flow_points[2])
            # shift domain to quadrant 1
            self.flow_points = (new_points[0]-new_points[0][0], new_points[1]-new_points[1][0],
                                new_points[2]-new_points[2][0])
            # update environment dimensions
            self.L = [self.flow_points[dim][-1] for dim in range(3)]
        else:
            # 2D shifting and environment updating
            self.flow_points = (new_points[0]-new_points[0][0], new_points[1]-new_points[1][0])
            self.L = [self.flow_points[dim][-1] for dim in range(2)]

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
            


    def move_swarms(self, dt=1.0, params=None, ib_collisions='sliding', 
                    silent=False):
        '''Move all swarms in the environment.

        Parameters
        ----------
        dt : float
            length of time step to move all agents
        params : any, optional
            parameters to pass along to get_positions, if necessary
        ib_collisions : {None, 'sliding' (default), 'sticky'}
            Type of interaction with immersed boundaries. If None, turn off all 
            interaction with immersed boundaries. In sliding collisions, 
            conduct recursive vector projection until the length of the original 
            vector is exhausted. In sticky collisions, just return the point of 
            intersection.
        silent : bool, default=False
            If True, suppress printing the updated time.
        '''

        for s in self.swarms:
            s.move(dt, params, ib_collisions, update_time=False)

        # update time
        self.time_history.append(self.time)
        self.time += dt
        if not silent:
            print('time = {}'.format(np.round(self.time,11)))



    def set_boundary_conditions(self, x_bndry, y_bndry, z_bndry=None):
        '''Check and set boundary conditions. Set Z-dimension only if
        zdim is not None. Each boundary condition must be either a list or an
        iterable of length 2.
        '''

        supprted_conds = ['zero', 'noflux', 'periodic']
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

        # If fCubicSplines do not exist, create them.
        if self.flow is None:
            raise RuntimeError("Cannot temporally interpolate None flow.")
        if not all([type(f) == fCubicSpline for f in self.flow]):
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
            return [f(time) for f in self.flow]



    def _create_temporal_interpolations(self):
        '''Create PPoly fCubicSplines to interpolate the fluid velocity in time.'''
        for n, flow in enumerate(self.flow):
            # Defaults to axis=0 along which data is varying, which is t axis
            # Defaults to not-a-knot boundary condition, resulting in first
            #   and second segments at curve ends being the same polynomial
            # Defaults to extrapolating out-of-bounds points based on first
            #   and last intervals. This will be overriden by this method
            #   to use constant extrapolation instead.
            if type(flow) != fCubicSpline:
                self.flow[n] = fCubicSpline(self.flow_times, flow)



    def _create_dt_interpolations(self):
        '''Create PPoly objects for dudt.'''
        self.dt_interp = []
        if not all([type(f) == fCubicSpline for f in self.flow]):
            self._create_temporal_interpolations()
        for ppoly in self.flow:
            self.dt_interp.append(ppoly.derivative())



    def interpolate_flow(self, positions, flow=None, flow_points=None, 
                         time=None, method='linear'):
        '''Spatially interpolate the fluid velocity field (or another flow field) 
        at the supplied positions. If flow is None and self.flow is time-varying,
        the flow field will be interpolated in time first, using the current 
        environmental time, or a different time if provided.

        Parameters
        ----------
        positions : array
            NxD locations at which to interpolate the flow field, where D is the 
            dimension of the system.
        flow : list-like of arrays, optional
            The fluid velocity data, with each array representing one spatial 
            component of the velocity vector. The first dimension of each array 
            is time if the fluid velocity field is time varying. If None, the 
            environmental flow field is used. Interpolated in time if necessary.
        flow_points : list-like of 1-D arrays, optional
            The set of coordinates along each dimension that defines the mesh 
            grid for the fluid velocity field. If None, the environmental flow 
            points is used.
        time : float, optional
            if None, the present time. Otherwise, the flow field will be
            interpolated to the time given based on the environment flow_times.
            This is only supported for environmental flow fields (not ones 
            passed in as an argument).
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

        if flow_points is None:
            flow_points = self.flow_points
        
        if method == 'splinef2d':
            raise RuntimeError('Extrapolation is not supported in splinef2d.'+
                               ' This is needed for RK4 solvers, and so is not'+
                               ' a supported method in interpolate_flow.')

        x_vel = interpolate.interpn(flow_points, flow[0],
                                    positions, method=method, 
                                    bounds_error=False, fill_value=None)
        y_vel = interpolate.interpn(flow_points, flow[1],
                                    positions, method=method, 
                                    bounds_error=False, fill_value=None)
        if len(flow) == 3:
            z_vel = interpolate.interpn(flow_points, flow[2],
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
        self.mag_grad = flow_grad
        self.mag_grad_time = self.time



    def define_pic_grid(self, positions, dx, dy, dz=None, return_neighbors=True):
        '''Creates a grid across the domain with cells of size dx by dy for use 
        in a particle-in-cell method. dx and dy should divide the domain evenly 
        in their respective directions and represent the furthest away one needs 
        to look from any individual agent in order to get all neighbor 
        interactions (e.g., a characteristic distance).

        Returns a dictionary of cells in which keys are (i,j) tuples indexing 
        the cells starting at zero from the origin, and the values they point to 
        are lists of agent indices whose positions are within that cell.

        If return_neighbors is True, will also return a dictionary of cell 
        indices in which the values are of agents located either within that 
        cell OR in a neighboring cell. Neighbor cells are the 8 cells vertically 
        or horizontally adjacent or diagonally adjacent (in 2D). Adjacency on the 
        boundaries of the domain depends upon the environment boundary condition: 
        zero or no-flux will treat the edge of the environment as a hard 
        boundary while periodic will wrap around to find neighboring cells.

        Parameters
        ----------
        positions : Nx2 or Nx3 ndarray
            ndarray of agent positions (e.g., swarm.positions)
        dx : float
            length of grid cell in the x-direction
        dy : float
            length of grid cell in the y-direction
        dz : float, optional
            length of grid cell in the z-direction
        return_neighbors : bool, default=True
            if True, return both a dictionary a dictionary of cells -> agent 
            indices located inside cell AND a dictionary of cells -> agent 
            indices in both cell and neighboring cells

        Returns
        -------
        cells dictionary, or tuple of two dictonaries (cells, neighbors)
        '''

        if dz is not None:
            DIM3 = True
        else:
            DIM3 = False

        Nx = round(self.L[0]/dx)
        Ny = round(self.L[1]/dy)
        if DIM3: Nz = round(self.L[2]/dz)

        assert np.isclose(Nx*dx,self.L[0]), "dx does not divide domain evenly."
        assert np.isclose(Ny*dy,self.L[1]), "dy does not divide domain evenly."
        if DIM3:
            assert np.isclose(Nz*dz,self.L[2]), "dz does not divide domain evenly."

        # Nx2 array of agent position indices
        if DIM3:
            pos_ind = (positions//np.array([dx,dy,dz])).astype(int)
        else:
            pos_ind = (positions//np.array([dx,dy])).astype(int)

        # Form a dictionary of cells
        cells = {}
        for ii in range(positions.shape[0]):
            if tuple(pos_ind[ii,:]) in cells:
                cells[tuple(pos_ind[ii,:])].append(ii)
            else:
                cells[tuple(pos_ind[ii,:])] = [ii]

        if return_neighbors:
            # Form a dictionary of all agents in the cell OR neighbors
            neigh = {}
            if not DIM3:
                for x in range(Nx):
                    for y in range(Ny):
                        if (x,y) in cells:
                            nearby_agents = []
                            # get list of adjacent cells depending on BC
                            idx_list = []
                            # x-1 column
                            if x-1<0 and self.bndry[0][0] == 'periodic':
                                if y-1<0 and self.bndry[1][0] == 'periodic':
                                    idx_list.append((Nx-1,Ny-1))
                                elif y-1>=0:
                                    idx_list.append((Nx-1,y-1))
                                idx_list.append((Nx-1,y))
                                if y+1==Ny and self.bndry[1][1] == 'periodic':
                                    idx_list.append((Nx-1,0))
                                elif y+1!=Ny:
                                    idx_list.append((Nx-1,y+1))
                            elif x-1>=0:
                                if y-1<0 and self.bndry[1][0] == 'periodic':
                                    idx_list.append((x-1,Ny-1))
                                elif y-1>=0:
                                    idx_list.append((x-1,y-1))
                                idx_list.append((x-1,y))
                                if y+1==Ny and self.bndry[1][1] == 'periodic':
                                    idx_list.append((x-1,0))
                                elif y+1!=Ny:
                                    idx_list.append((x-1,y+1))
                            # x column
                            if y-1<0 and self.bndry[1][0] == 'periodic':
                                idx_list.append((x,Ny-1))
                            elif y-1>=0:
                                idx_list.append((x,y-1))
                            idx_list.append((x,y))
                            if y+1==Ny and self.bndry[1][1] == 'periodic':
                                idx_list.append((x,0))
                            elif y+1!=Ny:
                                idx_list.append((x,y+1))
                            # x+1 column
                            if x+1==Nx and self.bndry[0][1] == 'periodic':
                                if y-1<0 and self.bndry[1][0] == 'periodic':
                                    idx_list.append((0,Ny-1))
                                elif y-1>=0:
                                    idx_list.append((0,y-1))
                                idx_list.append((0,y))
                                if y+1==Ny and self.bndry[1][1] == 'periodic':
                                    idx_list.append((0,0))
                                elif y+1!=Ny:
                                    idx_list.append((0,y+1))
                            elif x+1!=Nx:
                                if y-1<0 and self.bndry[1][0] == 'periodic':
                                    idx_list.append((x+1,Ny-1))
                                elif y-1>=0:
                                    idx_list.append((x+1,y-1))
                                idx_list.append((x+1,y))
                                if y+1==Ny and self.bndry[1][1] == 'periodic':
                                    idx_list.append((x+1,0))
                                elif y+1!=Ny:
                                    idx_list.append((x+1,y+1))

                            for idx in idx_list:
                                if idx in cells:
                                    nearby_agents += cells[idx]
                            neigh[(x,y)] = nearby_agents
            # 3D
            else:
                if self.bndry[0][0] == 'periodic':
                    x_list = [Nx-1]; xstrt = 1
                else:
                    x_list = []; xstrt = 0
                x_list += list(range(Nx))
                if self.bndry[0][1] == 'periodic':
                    x_list.append(0); xend = len(x_list)-1
                else:
                    xend = len(x_list)

                if self.bndry[1][0] == 'periodic':
                    y_list = [Ny-1]; ystrt = 1
                else:
                    y_list = []; ystrt = 0
                y_list += list(range(Ny))
                if self.bndry[1][1] == 'periodic':
                    y_list.append(0); yend = len(y_list)-1
                else:
                    yend = len(y_list)

                if self.bndry[2][0] == 'periodic':
                    z_list = [Nz-1]; zstrt = 1
                else:
                    z_list = []; zstrt = 0
                z_list += list(range(Nz))
                if self.bndry[2][1] == 'periodic':
                    z_list.append(0); zend = len(z_list)-1
                else:
                    zend = len(z_list)

                for xid in range(xstrt,xend):
                    for yid in range(ystrt,yend):
                        for zid in range(zstrt,zend):
                            if (x_list[xid],y_list[yid],z_list[zid]) in cells:
                                nearby_agents = []
                                # get list of adjacent cells depending on BC
                                idx_list = []
                                # x-1
                                if xid-1>=0:
                                    if yid-1>=0:
                                        if zid-1>=0:
                                            idx_list.append((x_list[xid-1],y_list[yid-1],z_list[zid-1]))
                                        idx_list.append((x_list[xid-1],y_list[yid-1],z_list[zid]))
                                        if zid+1<len(z_list):
                                            idx_list.append((x_list[xid-1],y_list[yid-1],z_list[zid+1]))
                                    
                                    if zid-1>=0:
                                        idx_list.append((x_list[xid-1],y_list[yid],z_list[zid-1]))
                                    idx_list.append((x_list[xid-1],y_list[yid],z_list[zid]))
                                    if zid+1<len(z_list):
                                        idx_list.append((x_list[xid-1],y_list[yid],z_list[zid+1]))

                                    if yid+1<len(y_list):
                                        if zid-1>=0:
                                            idx_list.append((x_list[xid-1],y_list[yid+1],z_list[zid-1]))
                                        idx_list.append((x_list[xid-1],y_list[yid+1],z_list[zid]))
                                        if zid+1<len(z_list):
                                            idx_list.append((x_list[xid-1],y_list[yid+1],z_list[zid+1]))

                                # x
                                if yid-1>=0:
                                    if zid-1>=0:
                                        idx_list.append((x_list[xid],y_list[yid-1],z_list[zid-1]))
                                    idx_list.append((x_list[xid],y_list[yid-1],z_list[zid]))
                                    if zid+1<len(z_list):
                                        idx_list.append((x_list[xid],y_list[yid-1],z_list[zid+1]))
                                
                                if zid-1>=0:
                                    idx_list.append((x_list[xid],y_list[yid],z_list[zid-1]))
                                idx_list.append((x_list[xid],y_list[yid],z_list[zid]))
                                if zid+1<len(z_list):
                                    idx_list.append((x_list[xid],y_list[yid],z_list[zid+1]))

                                if yid+1<len(y_list):
                                    if zid-1>=0:
                                        idx_list.append((x_list[xid],y_list[yid+1],z_list[zid-1]))
                                    idx_list.append((x_list[xid],y_list[yid+1],z_list[zid]))
                                    if zid+1<len(z_list):
                                        idx_list.append((x_list[xid],y_list[yid+1],z_list[zid+1]))

                                # x+1
                                if xid+1<len(x_list):
                                    if yid-1>=0:
                                        if zid-1>=0:
                                            idx_list.append((x_list[xid+1],y_list[yid-1],z_list[zid-1]))
                                        idx_list.append((x_list[xid+1],y_list[yid-1],z_list[zid]))
                                        if zid+1<len(z_list):
                                            idx_list.append((x_list[xid+1],y_list[yid-1],z_list[zid+1]))
                                    
                                    if zid-1>=0:
                                        idx_list.append((x_list[xid+1],y_list[yid],z_list[zid-1]))
                                    idx_list.append((x_list[xid+1],y_list[yid],z_list[zid]))
                                    if zid+1<len(z_list):
                                        idx_list.append((x_list[xid+1],y_list[yid],z_list[zid+1]))

                                    if yid+1<len(y_list):
                                        if zid-1>=0:
                                            idx_list.append((x_list[xid+1],y_list[yid+1],z_list[zid-1]))
                                        idx_list.append((x_list[xid+1],y_list[yid+1],z_list[zid]))
                                        if zid+1<len(z_list):
                                            idx_list.append((x_list[xid+1],y_list[yid+1],z_list[zid+1]))

                                for idx in idx_list:
                                    if idx in cells:
                                        nearby_agents += cells[idx]
                                neigh[(x_list[xid],y_list[yid],z_list[zid])] = nearby_agents

            return cells, neigh
        return cells 



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

        This method will set the following environment attributes:
        - environment.FTLE_largest
        - environment.FTLE_smallest
        - environment.FTLE_loc
        - environment.FTLE_t0
        - environment.FTLE_T
        - environment.FTLE_grid_dim

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
            integration time. Default is 1, but longer is better (up to a point),
            unless smallest FTLE is desired and agents are leaving the domain...
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
            generator. If unspecified, will default to the values for the first 
            agent in the props of the swarm provided.
        t_bound : float, optional
            if solving ode or tracer particles, this is the bound on
            the RK45 integration step size. Defaults to dt/100.
        swrm : swarm object, optional 
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
            # Get a shallow copy of the swarm passed in
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
            # Re-initialize props based off of the first agent
            new_props = pd.DataFrame()
            for prop in s.props:
                new_props[prop] = [s.props[prop][0] for ii in range(s.positions.shape[0])]
            s.props = new_props
            # Re-initialize ib_collision
            s.ib_collision = np.full(s.positions.shape[0], False)

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
                    if not ma.is_masked(s.positions):
                        y = s.positions
                    else:
                        y = s.positions[~s.positions[:,0].mask,:]
                else:
                    if not ma.is_masked(s.positions):
                        y = np.concatenate((s.positions, s.velocities))
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

                # Save current position to put in the history
                old_positions = s.positions.copy()
                old_velocities = s.velocities.copy()

                # pull solution into swarm object's position/velocity attributes
                if ode_gen is None:
                    s.positions[~ma.getmaskarray(s.positions[:,0]),:] = y_new
                else:
                    N = round(y_new.shape[0]/2)
                    s.positions[~ma.getmaskarray(s.positions[:,0]),:] = y_new[:N,:]
                    s.velocities[~s.velocities[:,0].mask,:] = y_new[N:,:]

                # Update history
                s.pos_history.append(old_positions)
                s.vel_history.append(old_velocities)
                # apply boundary conditions
                old_mask = s.positions.mask.copy()
                s.apply_boundary_conditions(dt)
                # copy time to non-masked locations
                last_time[~ma.getmaskarray(s.positions[:,0])] = new_time
                
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
                # Save current position to put in the history
                old_positions = s.positions.copy()
                old_velocities = s.velocities.copy()
                # Conditionally save props to put in the history too
                if s.props_history is not None:
                    old_props = s.props.copy()
                
                # Update positions
                s.positions[:,:] = s.get_positions(dt, params)

                # Update history
                s.pos_history.append(old_positions)
                s.vel_history.append(old_velocities)
                if self.props_history is not None:
                    s.props_history.append(old_props)

                # Update velocity and acceleration
                s.velocities[:,:] = (s.positions - old_positions)/dt
                s.accelerations[:,:] = (s.velocities - old_velocities)/dt
                # Apply boundary conditions.
                s.apply_boundary_conditions(dt)
                # Update time
                self.time_history.append(self.time)
                self.time += dt
                # copy time to non-masked locations
                last_time[~ma.getmaskarray(s.positions[:,0])] = self.time

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
        self.FTLE_loc_end = s.positions
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
            dx1 = self.flow_points[0][grid_loc[0]] - self.flow_points[0][neigh_list[0,0]]
            dx2 = self.flow_points[0][neigh_list[1,0]] - self.flow_points[0][grid_loc[0]]
            dy1 = self.flow_points[1][grid_loc[1]] - self.flow_points[1][neigh_list[2,1]]
            dy2 = self.flow_points[1][neigh_list[3,1]] - self.flow_points[1][grid_loc[1]]
            # central differencing
            dvydx = (v_y[tuple(neigh_list[1,:])]-v_y[tuple(neigh_list[0,:])])/(dx1+dx2)
            dvxdy = (v_x[tuple(neigh_list[3,:])]-v_x[tuple(neigh_list[2,:])])/(dy1+dy2)
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
                dataio.write_vtk_2D_rectilinear_grid_scalars(path, name, vort, self.L, cyc, time)
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
                dataio.write_vtk_2D_rectilinear_grid_scalars(path, out_name, vort, self.L, cyc, time)
        if time_history or not flow_times:
            vort = self.get_2D_vorticity(time=self.time)
            dataio.write_vtk_2D_rectilinear_grid_scalars(path, name, vort, self.L, cycle, self.time)



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
                dataio.write_vtk_rectilinear_grid_vectors(path, name, flow, self.L, cyc, time)
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
                dataio.write_vtk_rectilinear_grid_vectors(path, out_name, flow, self.L, cyc, time)
        if time_history or not flow_times:
            flow = self.interpolate_temporal_flow(time=self.time)
            dataio.write_vtk_rectilinear_grid_vectors(path, name, flow, self.L, cycle, self.time)



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
        List of ndarrays
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
            # If PPoly derivatives do not exist, create them.
            if self.dt_interp is None:
                self._create_dt_interpolations()
            return [dfdt(time) for dfdt in self.dt_interp]



    def calculate_DuDt(self, t_indx=None, time=None):
        '''Calculate and store material derivative of the fluid velocity with 
        respect to time. Defaults to interpolating at the current time, given by 
        self.time. Gradient is calculated via second order accurate central 
        differences (using numpy) with second order accuracy at the boundaries.
        The material derivative is saved in case it is needed again.

        The material derivative is given by
        .. math::
        \\frac{D\\mathbf{u}}{Dt} = \\mathbf{u}_t + 
        (\\nabla\\mathbf{u})\\mathbf{u}
        
        Parameters
        ----------
        t_indx : int, optional
            Interpolate at a time referred to by self.envir.time_history[t_indx]
        time : float, optional
            Interpolate at a specific time. default is current time.
        '''

        DIM3 = (len(self.L) == 3)

        TIME_DEP = len(self.flow[0].shape) != len(self.L)

        if DIM3:
            axis_tuple = (1,2,3)
        else:
            axis_tuple = (1,2)

        if not TIME_DEP:
            flow = self.flow
        else:
            # first, interpolate flow in time.
            flow = self.interpolate_temporal_flow(t_indx=t_indx, time=time)
            
        flow_grad = np.gradient(np.array(flow),
                    *self.flow_points, edge_order=2, axis=axis_tuple)

        # Take dot product
        DuDt = []
        for g,f in zip(flow_grad,flow):
            DuDt.append(g*f)
        DuDt = np.sum(DuDt, axis=0)

        # Add dudt
        DuDt += np.array(self.dudt(t_indx, time))

        self.DuDt = [u for u in DuDt]
        if t_indx is None and time is None:
            self.DuDt_time = self.time
        elif t_indx is not None:
            self.DuDt_time = self.time_history[t_indx]
        else:
            self.DuDt_time = time



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



    def regenerate_flow_data(self):
        '''Regenerates the original fluid velocity data from temporal spline 
        objects.
        '''
        for n, flow in enumerate(self.flow):
            self.flow[n] = flow.regenerate_data()



    def __reset_flow_variables(self, incl_rho_mu_U=False):
        '''To be used when the fluid flow changes. Resets all the helper
        parameters and reports new domain.'''

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
        print("Fluid updated. Planktos domain size is now {}.".format(self.L))



    def __reset_flow_deriv(self):
        '''Reset properties that are derived from the flow velocity.'''

        self.mag_grad = None
        self.mag_grad_time = None
        self.dt_interp = None
        self.DuDt = None
        self.DuDt_time = None



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
                axHistx.set_ylim(bottom=0)
                axHisty.set_ylim(ax.get_ylim())
                axHisty.set_xlim(left=0)
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
                    ax.axvline(x=g, ymax=self.h_p/self.L[1], color='.5', zorder=0.5)

            # plot any ghost structures
            for plot_func, args in zip(self.plot_structs, 
                                       self.plot_structs_args):
                if args is None:
                    plot_func(ax, args)
                else:
                    plot_func(ax, *args)

            # plot ibmesh
            if self.ibmesh is not None:
                line_segments = LineCollection(self.ibmesh)
                line_segments.set_color(self.ibmesh_color)
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
                ax = fig.add_subplot(projection='3d')
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
                ax = fig.add_subplot(121, projection='3d')
                ax.set_title('Organism positions')
                # No real solution to 3D aspect ratio...
                #ax.set_aspect('equal','box')

                # histograms
                int_ticks = MaxNLocator(nbins='auto', integer=True)
                axHistx = plt.axes(rect_histx)
                axHistx.set_xlim((0, self.L[0]))
                axHistx.set_ylim(bottom=0)
                axHistx.yaxis.set_major_locator(int_ticks)
                axHistx.set_ylabel('X    ', rotation=0)
                axHisty = plt.axes(rect_histy)
                axHisty.set_xlim((0, self.L[1]))
                axHisty.set_ylim(bottom=0)
                axHisty.yaxis.set_major_locator(int_ticks)
                axHisty.set_ylabel('Y    ', rotation=0)
                axHistz = plt.axes(rect_histz)
                axHistz.set_xlim((0, self.L[2]))
                axHistz.set_ylim(bottom=0)
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
                structures.set_color(self.ibmesh_color)
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

        if self.flow is None:
            self.plot_envir(figsize=figsize)
            return

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
            max_u = self.flow[0].max(); max_v = self.flow[1].max()
            max_mag = np.linalg.norm(np.array([max_u,max_v]))
            if len(self.L) == len(self.flow[0].shape) or t is not None:
                # Single-time plot.
                if loc is None:
                    ax.quiver(self.flow_points[0][::M], self.flow_points[1][::M],
                              self.flow[0][::M,::M].T, self.flow[1][::M,::M].T, 
                              scale=max_mag*5, **kwargs)
                else:
                    ax.quiver(self.flow_points[0][::M], self.flow_points[1][::M],
                              self.flow[0][loc][::M,::M].T,
                              self.flow[1][loc][::M,::M].T, 
                              scale=max_mag*5, **kwargs)
            else:
                # Animation plot
                # create quiver object
                quiver = ax.quiver(self.flow_points[0][::M], self.flow_points[1][::M], 
                                   self.flow[0][0][::M,::M].T,
                                   self.flow[1][0][::M,::M].T, 
                                   scale=max_mag*5, **kwargs)
                # textual info
                time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                    fontsize=12)
        ########## 3D Plot #########        
        else:
            x, y, z = np.meshgrid(self.flow_points[0][::M], self.flow_points[1][::M], 
                                  self.flow_points[2][::M], indexing='ij')
            max_u = self.flow[0].max(); max_v = self.flow[1].max()
            max_w = self.flow[2].max()
            max_mag = np.linalg.norm(np.array([max_u,max_v,max_w]))
            if len(self.L) == len(self.flow[0].shape) or t is not None:
                # Single-time plot.
                if loc is None:
                    ax.quiver(x,y,z,self.flow[0][::M,::M,::M],
                                    self.flow[1][::M,::M,::M],
                                    self.flow[2][::M,::M,::M], 
                                    **kwargs)
                else:
                    ax.quiver(x,y,z,self.flow[0][loc][::M,::M,::M],
                                    self.flow[1][loc][::M,::M,::M],
                                    self.flow[2][loc][::M,::M,::M],
                                    **kwargs)
            else:
                # Animation plot
                quiver = ax.quiver(x,y,z,self.flow[0][0][::M,::M,::M],
                                         self.flow[1][0][::M,::M,::M],
                                         self.flow[2][0][::M,::M,::M],
                                         **kwargs)
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
        

        if smallest:
            grid_x, grid_y = np.mgrid[0:self.L[0]:self.FTLE_grid_dim[0]*1j,
                                      0:self.L[1]:self.FTLE_grid_dim[1]*1j]
            valid_points_x = self.FTLE_loc_end[~self.FTLE_loc_end[:,0].mask,0]
            valid_points_y = self.FTLE_loc_end[~self.FTLE_loc_end[:,1].mask,1]
            valid_vals = FTLE.flatten()[~self.FTLE_loc_end[:,0].mask]
            grid_sFTLE = interpolate.griddata((valid_points_x,valid_points_y),
                                              valid_vals, (grid_x,grid_y),
                                              method='cubic')
            pcm = ax.imshow(grid_sFTLE.T, extent=(0,self.L[0],0,self.L[1]), origin='lower')
            plt.title('Negative smallest fwrd-time FTLE field, $t_0$={}, $\Delta t$={}.\n'.format(
                    self.FTLE_t0, self.FTLE_T)+
                    'Interpolated from {} out of {} starting points left in domain.'.format(
                        np.sum(~self.FTLE_loc_end[:,0].mask),self.FTLE_loc_end.shape[0]))
        else:
            grid_x = np.reshape(self.FTLE_loc[:,0].data, self.FTLE_grid_dim)
            grid_y = np.reshape(self.FTLE_loc[:,1].data, self.FTLE_grid_dim)
            pcm = ax.pcolormesh(grid_x, grid_y, FTLE, shading='gouraud', 
                                cmap='plasma')
            plt.title('Largest fwrd-time FTLE field, $t_0$={}, $\Delta t$={}.'.format(
                    self.FTLE_t0, self.FTLE_T))
        axbbox = ax.get_position().get_points()
        cbaxes = fig.add_axes([axbbox[1,0]+0.01, axbbox[0,1], 0.02, axbbox[1,1]-axbbox[0,1]])
        plt.colorbar(pcm, cax=cbaxes)
        plt.show()



class fCubicSpline(interpolate.CubicSpline):
    '''
    Extends Scipy's CubicSpline object to get info about original fluid data.
    '''

    def __init__(self, flow_times, flow, **kwargs):
        super(fCubicSpline, self).__init__(flow_times, flow, **kwargs)

        self.shape = flow.shape
        self.data_max = flow.max()
        self.data_min = flow.min()



    def __getitem__(self, pos):
        '''
        Allows indexing into the interpolator.
        '''
        if type(pos) == int:
            return self.__call__(self.x[pos])
        elif type(pos) == slice:
            start = pos.start; stop = pos.stop; step = pos.step
            if start is None: start = 0
            if stop is None: stop = len(self.x)
            if step is None: step = 1
            return np.stack([self.__call__(self.x[n]) for 
                             n in range(start,stop,step)])
        elif type(pos) == tuple:
            if type(pos[0]) == int:
                return self.__call__(self.x[pos[0]])[pos[1:]]
            elif type(pos[0]) == slice:
                start = pos[0].start; stop = pos[0].stop; step = pos[0].step
                if start is None: start = 0
                if stop is None: stop = len(self.x)
                if step is None: step = 1
                return np.stack([self.__call__(self.x[n])[pos[1:]] for 
                                 n in range(start,stop,step)])
            else:
                raise IndexError('Only integers or slices supported in fCubicSpline.')
        else:
            raise IndexError('Only integers or slices supported in fCubicSpline.')



    def __setitem__(self, pos, val):
        raise RuntimeError("Cannot assign to spline object. "+
                           "Use regenerate_data to recreate original data first.")



    def max(self):
        return self.data_max

    def min(self):
        return self.data_min

    def absmax(self):
        return np.abs(np.array([self.data_max, self.data_min])).max()

    def regenerate_data(self):
        '''
        Rebuild the original data.
        '''
        return np.stack([self.__call__(val) for val in self.x])


