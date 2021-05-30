#! /usr/bin/env python3

'''
Swarm class file, for simulating many individuals at once.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

import sys, os, warnings, copy
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
from pathlib import Path
from math import exp, log
from itertools import combinations
import numpy as np
import numpy.ma as ma
from scipy import interpolate, stats
from scipy.spatial import distance, ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from matplotlib import animation, colors
from matplotlib.collections import LineCollection
import motion
import data_IO

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class environment:

    def __init__(self, Lx=10, Ly=10, Lz=None,
                 x_bndry='zero', y_bndry='zero', z_bndry='noflux', flow=None,
                 flow_times=None, rho=None, mu=None, nu=None, char_L=None, 
                 U=None, init_swarms=None, units='m'):
        ''' Initialize environmental variables.

        Arguments:
            Lx: Length of domain in x direction, m
            Ly: Length of domain in y direction, m
            Lz: Length of domain in z direction, or None
            x_bndry: [left bndry condition, right bndry condition] (default: zero)
            y_bndry: [low bndry condition, high bndry condition] (default: zero)
            z_bndry: [low bndry condition, high bndry condition] (default: noflux)
            flow: [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k]),
                   z-vel field ndarray if 3D]
                Note! i is x index, j is y index, with the value of x/y increasing
                as the index increases. It is assumed that the flow mesh is equally 
                spaced and includes values on the domain bndry
                self.flow = None is a valid flow, recognized to have fluid
                velocity of zero everywhere.
            flow_times: [tstart, tend] or iterable of times at which flow is specified
                     or scalar dt; required if flow is time-dependent.
            rho: fluid density of environment, kg/m**3 (optional, m here meaning length units).
                Auto-calculated if mu and nu are provided.
            mu: dynamic viscosity, kg/(m*s), Pa*s, N*s/m**2 (optional, m here meaning length units).
                Auto-calculated if rho and nu are provided.
            nu: kinematic viscosity, m**2/s (optional, m here meaning length units).
                Auto-calculated if rho and mu are provided.
            char_L: characteristic length scale. Used for calculating Reynolds
                number, especially in the case of immersed structures (ibmesh)
                and/or when simulating inertial particles
            init_swarms: initial swarms in this environment
            units: length units to use, default is meters. Note that you will
                manually need to change self.g (accel due to gravity) if using
                something else.

        Other properties:
            flow_points: points defining the spatial grid for flow data
            fluid_domain_LLC: original lower-left corner of domain (if from data)
            tiling: (x,y) how much tiling was done in the x and y direction
            orig_L: length of the domain in x and y direction (Lx,Ly) before tiling occured
            h_p: height of porous region
            g: accel due to gravity (length units/s**2)
            struct_plots: List of functions that plot additional environment structures
            struct_plot_args: List of argument tuples to be passed to these functions, after ax
            time: current time
            time_history: list of past time states
            grad: gradient of flow magnitude
            grad_time: sim time at which gradient was calculated

        Right now, supported agent boundary conditions are 'zero' (default) and 'noflux'.
        '''

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
            self.rho = rho
            self.mu = mu
            self.nu = mu/rho
        elif rho is not None and nu is not None:
            self.rho = rho
            self.mu = nu*rho
            self.nu = nu
        elif mu is not None and nu is not None:
            self.rho = mu/nu
            self.mu = mu
            self.nu = nu
        else:
            self.rho = rho
            self.mu = mu
            self.nu = nu

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



    def set_brinkman_flow(self, alpha, h_p, U, dpdx, res=101, tspan=None):
        '''Specify fully developed 2D or 3D flow with a porous region.
        Velocity gradient is zero in the x-direction; all flow moves parallel to
        the x-axis. Porous region is the lower part of the y-domain (2D) or
        z-domain (3D) with width=a and an empty region above. For 3D flow, the
        velocity profile is the same on all slices y=c. The decision to set
        2D vs. 3D flow is based on the dimension of the current domain.

        Arguments:
            alpha: equal to 1/(hydraulic permeability). alpha=0 implies free flow (infinitely permeable)
            h_p: height of porous region
            U: velocity at top of domain (v in input3d). scalar or list-like.
            dpdx: dp/dx change in momentum constant. scalar or list-like.
            res: number of points at which to resolve the flow (int), including boundaries
            tspan: [tstart, tend] or iterable of times at which flow is specified
                if None and U/dpdx are iterable, dt=1 will be used.

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.h_p = h_p

        Calls:
            self.__set_flow_variables
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

        Arguments:
            tspan: [tstart, tend] or iterable of times at which flow is specified
                    or scalar dt. Required if flow is time-dependent; None will
                    be interpreted as non time-dependent flow.
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
        "Vegetated Open Channel Flow". The decision to set 2D vs. 3D flow is 
        based on the dimension of the domain and this flow is always
        time-independent by nature. The following parameters must be given:

        Arguments:
            a: vegetation density, given by Az*m, where Az is the frontal area
                of vegetation per unit depth and m the number of stems per unit area (1/m)
                (assumed constant)
            h_p: plant height (m)
            Cd: drag coefficient (assumed uniform) (unitless)
            S: bottom slope (unitless, 0-1 with 0 being no slope, resulting in no flow)
            res: number of points at which to resolve the flow (int), including boundaries

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.h_p = a

        Calls:
            self.__set_flow_variables
        '''
        # Get channel height
        H = self.L[-1]
        # Get empirical length scale, Meijer and Van Valezen (1999)
        alpha = 0.0144*np.sqrt(H*h_p)
        # dimensionless beta
        beta = np.sqrt(Cd*a*h_p**2/alpha)
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
        '''Apply flow within and above a uniform homogenous canopy according to the
        model described in Finnigan and Belcher (2004), 
        "Flow over a hill covered with a plant canopy". The decision to set 2D vs. 3D flow is 
        based on the dimension of the domain. Default values for beta and C are
        based on Finnigan & Belcher. Must specify two of u_star, U_h, and beta, 
        though beta has a default value of 0.3 so just giving u_star or U_h will also work. 
        If one of u_star, U_h, or beta is given as a list-like object, the flow will vary in time.

        Arguments:
            h: height of canopy (m)
            a: leaf area per unit volume of space m^{-1}. Typical values are
                a=1.0 for dense spruce to a=0.1 for open woodland
            u_star: canopy friction velocity. u_star = U_h*beta OR
            U_h: wind speed at top of canopy. U_h = u_star/beta
            beta: mass flux through the canopy (u_star/U_h)
            C: drag coefficient of indivudal canopy elements
            res: number of points at which to resolve the flow (int), including boundaries
            tspan: [tstart, tend] or iterable of times at which flow is specified
                if None and u_star, U_h, and/or beta are iterable, dt=1 will be used.

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.h_p = h

        Calls:
            self.__set_flow_variables
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



    def read_IB2d_vtk_data(self, path, dt, print_dump,
                           d_start=0, d_finish=None):
        '''Reads in vtk flow velocity data generated by IB2d and sets environment
        variables accordingly.

        Can read in vector data with filenames u.####.vtk or scalar data
        with filenames uX.####.vtk and uY.####.vtk.

        IB2d is an Immersed Boundary (IB) code for solving fully coupled
        fluid-structure interaction models in Python and MATLAB. The code is 
        hosted at https://github.com/nickabattista/IB2d

        Arguments:
            - path: path to folder with vtk data
            - dt: dt in input2d
            - print_dump: print_dump in input2d
            - d_start: number of first vtk dump to read in
            - d_finish: number of last vtk dump to read in, or None to read to end
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
                uX, uY, x, y = data_IO.read_2DEulerian_Data_From_vtk(path, numSim,
                                                                     strChoice,xy)
                X_vel.append(uX.T) # (y,x) -> (x,y) coordinates
                Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates
            else:
                # read in x-directed Velocity Magnitude #
                strChoice = 'uX'; xy = True
                uX,x,y = data_IO.read_2DEulerian_Data_From_vtk(path,numSim,
                                                               strChoice,xy)
                X_vel.append(uX.T) # (y,x) -> (x,y) coordinates

                # read in y-directed Velocity Magnitude #
                strChoice = 'uY'
                uY = data_IO.read_2DEulerian_Data_From_vtk(path,numSim,
                                                           strChoice)
                Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates

            ###### The following is just for reference! ######

            # read in Vorticity #
            # strChoice = 'Omega'; first = 0
            # Omega = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                               strChoice,first)
            # read in Pressure #
            # strChoice = 'P'; first = 0
            # P = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                           strChoice,first)
            # read in Velocity Magnitude #
            # strChoice = 'uMag'; first = 0
            # uMag = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                              strChoice,first)
            # read in x-directed Forces #
            # strChoice = 'Fx'; first = 0
            # Fx = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                            strChoice,first)
            # read in y-directed Forces #
            # strChoice = 'Fy'; first = 0
            # Fy = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
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
        self.__reset_flow_variables(incl_rho_mu_U=True)
        self.reset()



    def read_IBAMR3d_vtk_data(self, filename):
        '''Reads in vtk flow data from a single source and sets environment
        variables accordingly. The resulting flow will be time invarient. It is
        assumed this data is a rectilinear grid.

        All environment variables will be reset.

        Arguments:
            filename: filename of data to read
        '''
        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(filename))

        data, mesh, time = data_IO.read_vtk_Rectilinear_Grid_Vector(filename)

        self.flow = list(data)
        self.flow_times = None

        # shift domain to quadrant 1
        self.flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                            mesh[2]-mesh[2][0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(3)]
        self.__reset_flow_variables(incl_rho_mu_U=True)
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

        Arguments:
            path: path to vtk data
            start: vtk file number to start with. If None, start at first one.
            finish: vtk file number to end with. If None, end with last one.
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
            data, mesh, time = data_IO.read_vtk_Rectilinear_Grid_Vector(str(this_file))
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
        self.__reset_flow_variables(incl_rho_mu_U=True)
        # record the original lower left corner (can be useful for later imports)
        self.fluid_domain_LLC = (mesh[0][0], mesh[1][0], mesh[2][0])
        # reset time
        self.reset()



    def read_npy_vtk_data(self, path, prefix='', flow_file='flow.npz',
                              flow_times_file='flow_times.npy',
                              flow_points_file='flow_points.npz',
                              flow_LLC_file='flow_LLC.npy'):
        '''Reads in numpy vtk flow data generated by IB2d and sets environment
        variables accordingly.

        Arguments:
            prefix: a prefix to add to each of the file names (instead of
                specifying each one individually)
            flow_file: string location/name of flow npz file
            flow_times_file: string location/name of flow_times npy file
            flow_points_file: string location/name of flow_points npz file
            flow_LLC_file: string location/name of LLC domain info, used in 3D only
        '''
        path = Path(path)
        if not path.is_dir(): 
            raise FileNotFoundError("Directory {} not found!".format(str(path)))
        flow_data = np.load(str(path/(prefix+flow_file)))
        self.flow = [flow_data['xflow'], flow_data['yflow'], flow_data['zflow']]
        flow_data.close()
        flow_times_ary = np.load(str(path/(prefix+flow_times_file)))
        if len(flow_times_ary.shape) > 0:
            self.flow_times = flow_times_ary
        else:
            self.flow_times = None
        flow_points_data = np.load(str(path/(prefix+flow_points_file)))
        if 'zpoints' in flow_points_data:
            self.flow_points = (flow_points_data['xpoints'], flow_points_data['ypoints'],
                                flow_points_data['zpoints'])
        else:
            self.flow_points = (flow_points_data['xpoints'], flow_points_data['ypoints'])
        flow_points_data.close()
        if len(self.flow) == 3:
            self.fluid_domain_LLC = tuple(np.load(str(path/(prefix+flow_LLC_file))))

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(len(self.flow))]
        self.__reset_flow_variables(incl_rho_mu_U=True)
        self.reset()



    def read_comsol_vtu_data(self, filename, vel_conv=None, grid_conv=None):
        '''Reads in vtu flow data from a single source and sets environment
        variables accordingly. The resulting flow will be time invarient.
        It is assumed this data is on a regular grid and that a grid section
        is included in the data.

        FOR NOW, THIS IS TIME INVARIANT ONLY.

        All environment variables will be reset.

        Arguments:
            filename: filename of data to read, incl. file extension
            vel_conv: scalar to multiply the velocity by in order to convert units
            grid_conv: scalar to multiply the grid by in order to convert units
        '''
        path = Path(filename)
        if not path.is_file(): 
            raise FileNotFoundError("File {} not found!".format(str(filename)))

        data, mesh = data_IO.read_vtu_mesh_velocity(filename)

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
        self.__reset_flow_variables(incl_rho_mu_U=True)
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

        ibmesh, self.max_meshpt_dist = data_IO.read_stl_mesh(filename)

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

        vertices = data_IO.read_IB2d_vertices(filename)
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
            intersections = swarm._seg_intersect_2D(seg[0,:], seg[1,:], 
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
            points, bounds = data_IO.read_vtk_Unstructured_Grid_Points(filename)
        elif filename.strip()[-7:] == '.vertex':
            points = data_IO.read_IB2d_vertices(filename)
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

        Arguments:
            x: number of tiles in the x direction (counting the one already there)
            y: number of tiles in the y direction (counting the one already there)
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

        Arguments:
            x_minus: number of times to duplicate bndry in the x- direction
            x_plus: number of times to duplicate bndry in the x+ direction
            y_minus: number of times to duplicate bndry in the y- direction
            y_plus: number of times to duplicate bndry in the y+ direction
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



    def add_swarm(self, swarm_s=100, init='random', seed=None, **kwargs):
        ''' Adds a swarm into this environment.

        Arguments:
            swarm_s: swarm object or size of the swarm (int). If a swarm object
                is given, the following arguments will be ignored (since the 
                object is already initialized)
            init: Method for initializing positions.
                Accepts 'random', 1D array for a single point, or a 2D array 
                to specify all points
            seed: Seed for random number generator
            kwargs: keyword arguments to be set as swarm properties
                (see swarm class for details)
        '''

        if isinstance(swarm_s, swarm):
            swarm_s._change_envir(self)
        else:
            return swarm(swarm_s, self, init=init, seed=seed, **kwargs)
            


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

        Arguments:
            t_indx: Interpolate at a time referred to by
                self.envir.time_history[t_indx]
            time: Interpolate at a specific time

        Returns:
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

        Arguments:
            positions: NxD locations at which to interpolate the flow field,
                where D is the dimension of the system.
            flow: if None, the environmental flow field. interpolated in time
                if necessary.
            time: if None, the present time. Otherwise, the flow field will be
                interpolated to the time given.
            method: spatial interpolation method to be passed to 
                scipy.interpolate.interpn. Anything but splinef2d is supported.'''

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



    def calculate_FTLE(self, grid_dim=None, testdir=None, t0=0, T=0.1, dt=0.001, 
                       ode=None, t_bound=None, swrm=None, params=None):
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

        Arguments:
            grid_dim: tuple of integers denoting the size of the grid in each
                dimension (x, y, [z]). Defaults to the fluid grid.
            testdir: grid points can heuristically be removed from the interior 
                of immersed structures. To accomplish this, a line will be drawn 
                from each point to a domain boundary. If the number of intersections
                is odd, the point is considered interior and masked. See grid_init 
                for details - this argument sets the direction that the line 
                will be drawn in (e.g. 'x1' for positive x-direction, 'y0' for 
                negative y-direction). If None, do not perform this check and use 
                all gridpoints.
            t0: start time for calculating FTLE (float). If None, default 
                behavior is to set t0=0.
                TODO: Interable to calculate at many times. Default then becomes 
                t0=0 for time invariant flows and calculate FTLE at all times 
                the flow field was specified at (self.flow_times) 
                for time varying flows?
            T: integration time (float). Default is 1, but longer is better 
                (up to a point).
            dt: if solving ode or tracer particles, this is the time step for 
                checking boundary conditions. If passing in a swarm object, 
                this argument represents the length of the Euler time steps.
            t_bound: if solving ode or tracer particles, this is the bound on
                the RK45 integration step size. Defaults to dt/100.
            ode: [optional] function handle for an ode to be solved specifying 
                deterministic equations of motion. Should have call signature 
                ODEs(t,x), where t is the current time (float) and x is a 2*NxD 
                array with the first N rows giving v=dxdt and the second N rows 
                giving dvdt. D is the spatial dimension of the problem. See the 
                ODE generator functions in motion.py for examples of valid functions. 
                The passed in ODEs will be solved using a grid of initial conditions 
                with Runge-Kutta.
            swarm: [optional] swarm object with user-defined movement rules as 
                specified by the get_positions method. This allows for arbitrary 
                FTLE calculations through subclassing and overriding this method. 
                Steps of length dt will be taken until the integration length T 
                is reached. The swarm object itself will not be altered; a shallow 
                copy will be created for the purpose of calculating the FTLE on 
                a grid.
            params: [optional] params to be passed to supplied swarm object's
                get_positions method.
            
        If both ode and swarm arguments are None, the default is to calculate the 
        FTLE based on massless tracer particles.

        Returns:
            swarm object: used to calculuate the FTLE
            list: list of dt integration steps
            ndarray: array the same size as the point grid giving the last time
                in the integration before each point exited the domain
        '''

        warnings.warn("FTLE requires more testing before it should be trusted!", UserWarning)

        ###########################################################
        ######              Setup swarm object               ######
        ###########################################################
        if grid_dim is None:
            grid_dim = tuple(len(pts) for pts in self.flow_points)

        if swrm is None:
            s = swarm(envir=self, init='grid', grid_dim=grid_dim, testdir=testdir)
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
            if ode is None:
                ode_fun = motion.tracer_particles(s, incl_dvdt=False)
                print("Finding {}D FTLE based on tracer particles.".format(len(grid_dim)))
            else:
                ode_fun = ode
                print("Finding {}D FTLE based on supplied ODE.".format(len(grid_dim)))
            print(prnt_str)

            # keep a list of all times solved for 
            #   (time history normally stored in environment class)
            current_time = t0
            time_list = [] 

            ### SOLVE ###
            while current_time < T:
                new_time = min(current_time + dt,T)
                ### TODO: REDO THIS SOLVER!!!!!!!!
                if ode is None:
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
                if ode is None:
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
                FTLE_largest[grid_loc] = np.log(np.sqrt(w[-1]))/T
                # FTLE_largest[grid_loc] = np.log(np.sqrt(w[-1]))/(t_calc-t0)
            if w[0] <= 0:
                FTLE_smallest[grid_loc] = ma.masked
            else:
                FTLE_smallest[grid_loc] = np.log(np.sqrt(w[0]))/T
                # FTLE_smallest[grid_loc] = np.log(np.sqrt(w[0]))/(t_calc-t0)

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

        Arguments:
            t_indx: integer ime index into self.envir.time_history[t_indx]
            time: time, float
            t_n: integer time index into self.flow_times[t_n]

        If all time arguments are None but the flow is time-varying, the vorticity
        at the current time will be returned. If more than one time argument is
        specified, only the first will be used.

        Returns:
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

        Arguments:
            path: string, location to save file(s)
            name: string, prefix name for file(s)
            time_history: if True, save vorticity data for each time step in the
                simulation history. Only for time-varying fluid.
            flow_times: if True, save vorticity data for each time at which the
                fluid velocity data is explicitly specified.
        '''

        if time_history:
            for cyc, time in enumerate(self.time_history):
                vort = self.get_2D_vorticity(t_n=cyc)
                data_IO.write_vtk_2D_uniform_grid_scalars(path, name, vort, self.L, cyc, time)
            cycle = len(self.time_history)
        else:
            cycle = None
        if flow_times:
            if time_history:
                out_name = 'omega'
            else:
                out_name = name
            for cyc, time in enumerate(self.flow_times):
                vort = self.get_2D_vorticity(t_indx=cyc)
                data_IO.write_vtk_2D_uniform_grid_scalars(path, out_name, vort, self.L, cyc, time)
        if time_history or not flow_times:
            vort = self.get_2D_vorticity(self.time)
            data_IO.write_vtk_2D_uniform_grid_scalars(path, name, vort, self.L, cycle, self.time)



    @property
    def Re(self):
        '''Return the Reynolds number at the current time based on mean fluid
        speed. Must have set self.char_L and self.nu'''

        return self.U*self.char_L/self.nu



    def dudt(self, t_indx=None, time=None):
        '''Return the derivative of the fluid velocity with respect to time.
        Defaults to interpolating at the current time, given by self.time.

        Arguments:
            t_indx: Interpolate at a time referred to by
                self.envir.time_history[t_indx]
            time: Interpolate at a specific time. default is current time.

        Returns:
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

        Arguments:
            smallest: If true, plot the negative, smallest, forward-time FTLE as 
                a way of identifying attracting Lagrangian Coherent Structures (see 
                Haller and Sapsis 2011). Otherwise, plot the largest, forward-time 
                FTLE as a way of identifying ridges (separatrix) of LCSs.
            clip_l: lower clip value (below this value, mask points)
            clip_h: upper clip value (above this value, mask points)
        '''

        warnings.warn("FTLE requires more testing before it should be trusted!", UserWarning)

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



class swarm:

    def __init__(self, swarm_size=100, envir=None, init='random', seed=None, 
                 shared_props=None, props=None, **kwargs):
        ''' Initializes planktos swarm in an environment.

        Arguments:
            swarm_size: Size of the swarm (int). ignored for 'grid' init method.
            envir: environment for the swarm, defaults to the standard environment
            init: Method for initalizing positions. See below.
            seed: Seed for random number generator, int or None
            shared_props: dictionary of properties shared by all agents
            props: Pandas dataframe of individual agent properties
            kwargs: keyword arguments to be set as a swarm property. They can
                be floats, ndarrays, or iterables, but keep in mind that
                problems will result with parsing if the number of agents is
                equal to the spatial dimension - this is to be avoided.
                Example key word arguments to include are:  
                diam -- diameter of the particles
                m -- mass of the particles
                Cd -- drag coefficient of the particles
                cross_sec -- cross-sectional area of the particles
                R -- density ratio

        Methods for initializing the swarm positions:
            - 'random': Uniform random distribution throughout the domain
            - 'grid': Uniform grid on interior of the domain, including capability
                to leave out closed immersed structures. In this case, swarm_size 
                is ignored since it is determined by the grid dimensions.
                Requires the additional keyword parameters:
                grid_dim = tuple of number of grid points in x, y, [and z] directions
                testdir: (optional) two character string for testing if points 
                    are in the interior of an immersed structure and if so, masking
                    them. The first char is x,y, or z denoting the dimensional direction
                    of the search ray, the second is either 0 or 1 denoting the 
                    direction (backward vs. forward) along that direction. See 
                    documentation of swarm.grid_init for more information.
            - 1D array-like: All positions set to a single point.
            - 2D array: All positions as specified. Shape NxD, D=dim of space.
                In this case, swarm_size is ignored.
        
        Initial agent velocities will be set as the local fluid velocity if present,
        otherwise zero. Assign to self.velocities to set your own.

        To customize agent behavior, subclass this class and re-implement the
        method get_positions (do not change the call signature).
        '''

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
        self.apply_boundary_conditions(no_ib=True)

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
        '''Return a grid of initial positions, potentially masking any grid 
        points in the interior of a closed, immersed structure. The full, unmasked 
        grid will be x_num by y_num [by z_num] on the interior and boundaries of 
        the domain.
        
        The output of this method is appropriate for finding FTLE.

        Grid list moves in the [Z direction], Y direction, then X direction.

        Arguments:
            x_num, y_num, [z_num]: number of grid points in each direction
            testdir: to check if a point is an interior to an immersed structure, 
                a line will be drawn from the point to a domain boundary. If the number 
                of immersed boundary intersections is odd, the point will be 
                considered interior and masked. This check will not be run at all 
                if testdir is None. Otherwise, specify a direction with one of 
                the following: 'x0','x1','y0','y1','z0','z1' (the last two for 
                3D problems only) denoting the dimension (x,y, or z) and the 
                direction (0 for negative, 1 for positive).

        NOTE: This algorithm is meant as a huristic only! It is not guaranteed 
        to mask all interior grid points, and will mask non-interior points if 
        there is not a clear line from the point to one of the boundaries of the 
        domain. If this method fails for your geometry and better accuracy is 
        needed, use this method as a starting point and mask/unmask as necessary.
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
        '''Save the full position history (with mask and time stamps), current
        velocity and acceleration to csv files. Save shared_props to a npz file 
        (since it is likely to contain some mixture of scalars and arrays, but
        does not vary between the agents so is less likely to be loaded outside
        of Python) and save props to json (since it is likely to contain a 
        variety of types of data, may need to be loaded outside of Python, 
        and json will be human readable).

        Arguments:
            path: string, directory for storing data
            name: string, prefix name for data files
            pos_fmt: format for storing position, vel, and accel data
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
        The format is as follows:
        The first row contains cycle and time information. The cycle is given, 
            and then each time stamp is repeated D times, where D is the spatial 
            dimension of the system.
        Each subsequent row corresponds to a different agent in the swarm.
        Reading across the columns of an agent row: first, a boolean is given
            showing the state of the mask for that time step. Agents are masked
            when they have exited the domain. Then, the position vector is given
            as a group of D columns for the x, y, (and z) direction. Each set
            of 1+D columns then corresponds to a different cycle/time, as 
            labeled by the values in the first row.
        The result is a csv that is N+1 by (1+D)*T, where N is the number of 
            agents, D is the dimension of the system, and T is the number of 
            times recorded.

        Arguments:
            filename: (string) path/name of the file to save the data to
            fmt: fmt argument to be passed to numpy.savetxt for position data
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
        just one file if all is False.

        Arguments:
            path: string, location to save the data
            name: string, name of dataset
            all: bool. if True, save the entire history including the current
                time. If false, save only the current time.
        '''
        if len(self.envir.L) == 2:
            DIM2 = True
        else:
            DIM2 = False

        if not all or len(self.envir.time_history) == 0:
            if DIM2:
                data = np.zeros((self.positions[~self.positions[:,0].mask,:].shape[0],3))
                data[:,:2] = self.positions[~self.positions[:,0].mask,:]
                data_IO.write_vtk_point_data(path, name, data)
            else:
                data_IO.write_vtk_point_data(path, name, self.positions[~self.positions[:,0].mask,:])
        else:
            for cyc, time in enumerate(self.envir.time_history):
                if DIM2:
                    data = np.zeros((self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:].shape[0],3))
                    data[:,:2] = self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:]
                    data_IO.write_vtk_point_data(path, name, data, 
                                                 cycle=cyc, time=time)
                else:
                    data_IO.write_vtk_point_data(path, name, 
                        self.pos_history[cyc][~self.pos_history[cyc][:,0].mask,:], 
                        cycle=cyc, time=time)
            cyc = len(self.envir.time_history)
            if DIM2:
                data = np.zeros((self.positions[~self.positions[:,0].mask,:].shape[0],3))
                data[:,:2] = self.positions[~self.positions[:,0].mask,:]
                data_IO.write_vtk_point_data(path, name, data, cycle=cyc,
                                             time=self.envir.time)
            else:
                data_IO.write_vtk_point_data(path, name, 
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



    def calc_re(self, u):
        '''Calculate Reynolds number as experienced by the swarm based on 
        environment variables and given flow speed, u, and diam (in shared_props)'''

        if self.envir.rho is not None and self.envir.mu is not None and\
            'diam' in self.shared_props:
            return self.envir.rho*u*self.shared_props['diam']/self.envir.mu
        else:
            raise RuntimeError("Parameters necessary for Re calculation are undefined.")



    def move(self, dt=1.0, params=None, update_time=True):
        '''Move all organisms in the swarm over an amount of time dt.
        Do not override this method when subclassing - override get_positions
        instead!

        Arguments:
            dt: time-step for move
            params: parameters to pass along to get_positions, if necessary
            update_time: whether or not to update the environment's time by dt
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
            self.apply_boundary_conditions()

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
        '''Returns all agent positions after a time step of dt.

        THIS IS THE METHOD TO OVERRIDE IF YOU WANT DIFFERENT MOVEMENT!
        NOTE: Do not change the call signature.

        This method must return the new positions of all agents following a time 
        step of length dt, whether due to behavior, drift, or anything else. 

        In this default implementation, movement is a random walk with drift
        as given by an Euler step solver of the appropriate SDE for this process.
        Drift is the local fluid velocity plus self.get_prop('mu'), and the
        stochasticity is determined by the covariance matrix self.get_prop('cov').

        NOTE: self.velocities and self.accelerations will automatically be updated
        outside of this method using finite differences.

        Arguments:
            dt: length of time step
            params: any other parameters necessary (optional)

        Returns:
            new agent positions
        '''

        # default behavior for Euler_brownian_motion is dift due to mu property
        #   plus local fluid velocity and diffusion given by cov property
        #   specifying the covariance matrix.
        return motion.Euler_brownian_motion(self, dt)



    def get_prop(self, prop_name):
        '''Return the property requested as either a scalar (if shared) or a 
        numpy array, ready for use in vectorized operations (left-most index
        specifies the agent).
        
        Arguments:
            prop_name: name of the property to return
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
        when adding a new one.'''
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
        '''

        if positions is None:
            positions = self.positions

        return self.envir.interpolate_flow(positions, self.envir.dudt(time=time), 
                                           method='linear')



    def get_fluid_gradient(self, positions=None):
        '''Return the gradient of the magnitude of the fluid velocity at all
        agent positions (or at provided positions) via linear interpolation of 
        the gradient. Gradient is calculated via second order accurate central 
        differences with second order accuracy at the boundaries and saved in 
        case it is needed again.
        The gradient is linearly interpolated from the fluid grid to the
        agent locations.
        '''

        if positions is None:
            positions = self.positions

        TIME_DEP = len(self.envir.flow[0].shape) != len(self.envir.L)
        flow_grad = None

        # If available, use the already calculuated gradient
        if self.envir.grad is not None:
            if not TIME_DEP:
                flow_grad = self.envir.grad
            elif self.envir.grad_time == self.envir.time:
                flow_grad = self.envir.grad

        # Otherwise, calculate the gradient
        if flow_grad is None:
            if not TIME_DEP:
                flow_grad = np.gradient(np.sqrt(
                                np.sum(np.array(self.envir.flow)**2, axis=0)
                                ), *self.envir.flow_points, edge_order=2)
            else:
                # first, interpolate flow in time. Then calculate gradient.
                flow_grad = np.gradient(
                                np.sqrt(np.sum(
                                np.array(self.envir.interpolate_temporal_flow())**2,
                                axis=0)), *self.envir.flow_points, edge_order=2)
            # save the newly calculuate gradient
            self.envir.grad = flow_grad
            self.envir.grad_time = self.envir.time

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



    def apply_boundary_conditions(self, no_ib=False):
        '''Apply boundary conditions to self.positions.
        
        For no flux, projections are really simple in this case (box), so we 
        just do them directly/manually.'''

        # internal mesh boundaries go first
        if self.envir.ibmesh is not None and not no_ib:
            # if there is no mask, loop over all agents, appyling internal BC
            # loop over (non-masked) agents, applying internal BC
            if np.any(self.positions.mask):
                for n, startpt, endpt in \
                    zip(np.arange(self.positions.shape[0])[~self.positions.mask[:,0]],
                        self.pos_history[-1][~self.positions.mask[:,0],:],
                        self.positions[~self.positions.mask[:,0],:]
                        ):
                    new_loc = self._apply_internal_BC(startpt, endpt, 
                                self.envir.ibmesh, self.envir.max_meshpt_dist)
                    ### DEBUGGING: check the new location before assignment ###
                    # this only works for the ib2d_ibmesh example for channel flow
                    # if not 2.5e-02 <= new_loc[1] < 2.25e-01 and not\
                    #     np.all(self.envir.Dhull.find_simplex(new_loc) < 0):
                    #     import pdb; pdb.set_trace()
                    #     new_loc = self._apply_internal_BC(startpt, endpt, 
                    #             self.envir.ibmesh, self.envir.max_meshpt_dist)
                    ### 3D DEBUGGING: check the new location before assignment ###
                    # if not np.all(self.envir.Dhull.find_simplex(new_loc) < 0):
                    #     import pdb; pdb.set_trace()
                    #     new_loc = self._apply_internal_BC(startpt, endpt, 
                    #             self.envir.ibmesh, self.envir.max_meshpt_dist)
                    self.positions[n] = new_loc
            else:
                for n in range(self.positions.shape[0]):
                    startpt = self.pos_history[-1][n,:]
                    endpt = self.positions[n,:]
                    new_loc = self._apply_internal_BC(startpt, endpt,
                                self.envir.ibmesh, self.envir.max_meshpt_dist)
                    ### DEBUGGING: check the new location before assignment ###
                    # this only works for the ib2d_ibmesh example for channel flow
                    # if not 2.5e-02 <= new_loc[1] < 2.25e-01 and not\
                    #     np.all(self.envir.Dhull.find_simplex(new_loc) < 0):
                    #     import pdb; pdb.set_trace()
                    #     new_loc = self._apply_internal_BC(startpt, endpt, 
                    #             self.envir.ibmesh, self.envir.max_meshpt_dist)
                    ### 3D DEBUGGING: check the new location before assignment ###
                    # if not np.all(self.envir.Dhull.find_simplex(new_loc) < 0):
                    #     import pdb; pdb.set_trace()
                    #     new_loc = self._apply_internal_BC(startpt, endpt, 
                    #             self.envir.ibmesh, self.envir.max_meshpt_dist)
                    self.positions[n] = new_loc

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
                           old_intersection=None, kill=False):
        '''Apply internal boundaries to a trajectory starting and ending at
        startpt and endpt, returning a new endpt (or the original one) as
        appropriate.

        Arguments:
            startpt: start location for agent trajectory (len 2 or 3 ndarray)
            endpt: end location for agent trajectory (len 2 or 3 ndarray)
            mesh: Nx2x2 or Nx3x3 array of eligible mesh elements
            max_meshpt_dist: max distance between two points on a mesh element
                (used to determine how far away from startpt to search for
                mesh elements)
            old_intersection: for internal use only, to check if we are
                bouncing back and forth between two boundaries as a result of
                a concave angle and the right kind of trajectory vector.
            kill: set to True in 3D case if we have previously slid along the
                boundary line between two mesh elements. This prevents such
                a thing from happening more than once, in case of pathological
                cases.

        Returns:
            newendpt: new end location for agent trajectory
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
        
        # If we do have an intersection, project remaining piece of vector
        #   onto mesh and repeat processes as necessary until we have a final
        #   result.
        return swarm._project_and_slide(startpt, endpt, intersection,
                                        close_mesh, max_meshpt_dist, DIM,
                                        old_intersection, kill)



    @staticmethod
    def _project_and_slide(startpt, endpt, intersection, mesh, max_meshpt_dist,
                           DIM, old_intersection=None, kill=False):
        '''Once we have an intersection, project and slide the remaining movement,
        and determine what happens if we fall off the edge of the simplex in
        all angle cases.

        Arguments:
            startpt: original start point of movement, before intersection
            endpt: original end point of movement, after intersection
            intersection: result of _seg_intersect_2D or _seg_intersect_3D_triangles
            mesh: Nx2x2 or Nx3x3 array of eligible mesh elements
            max_meshpt_dist: max distance between two points on a mesh element
                (used to determine how far away from startpt to search for
                mesh elements) - for passthrough to possible recursion
            DIM: dimension of system, either 2 or 3
            old_intersection: for internal use only, to check if we are
                bouncing back and forth between two boundaries as a result of
                a concave angle and the right kind of trajectory vector.
            kill: set to True in 3D case if we have previously slid along the
                boundary line between two mesh elements. This prevents such
                a thing from happening more than once, in case of pathological
                cases.

        Returns:
            newendpt: new endpoint for movement
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
        '''Find the intersection between two line segments, P and Q, returning
        None if there isn't one or if they are parallel.
        If Q is a 2D array, loop over the rows of Q finding all intersections
        between P and each row of Q, but only return the closest intersection
        to P0 (if there is one, otherwise None)

        This works for both 2D problems and problems in which P is a 3D segment
        roughly on a plane. The plane is described by the first two vectors Q, so
        in this case, Q0_list and Q1_list must have at two rows. The 3D problem
        is robust to cases where P is not exactly in the plane because the
        algorithm is actually checking to see if its projection onto the
        triangle crosses any of the lines in Q.

        This algorithm uses a parameteric equation approach for speed, based on
        http://geomalgorithms.com/a05-_intersect-1.html
        
        Arguments:
            P0: length 2 (or 3) array, first point in line segment P
            P1: length 2 (or 3) array, second point in line segment P 
            Q0_list: Nx2 (Nx3) ndarray of first points in a list of line segments.
            Q1_list: Nx2 (Nx3) ndarray of second points in a list of line segments.
            get_all: Return all intersections instead of just the first one.

        Returns:
            None if there is no intersection. Otherwise:
            x: length 2 (or 3) array giving the coordinates of the point of first 
               intersection
            s_I: the fraction of the line segment traveled from P0 before
                intersection occurred
            vec: directional unit vector of boundary (Q) intersected
            Q0: first endpoint of mesh segment intersected
            Q1: second endpoint of mesh segment intersected
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
        triangles given by Q0, Q1, Q2 where each row is a different triangle.
        Returns None if there is no intersection.

        This algorithm uses a parameteric equation approach for speed, based on
        http://geomalgorithms.com/a05-_intersect-1.html

        Arguments:
            P0: length 3 array, first point in line segment P
            P1: length 3 array, second point in line segment P 
            Q0: Nx3 ndarray of first points in a list of triangles.
            Q1: Nx3 ndarray of second points in a list of triangles.
            Q2: Nx3 ndarray of third points in a list of triangles.
            get_all: Return all intersections instead of just the first one.

        Returns:
            x: length 3 array giving the coordinates of the first point of intersection
            s_I: the fraction of the line segment traveled from P0 before
                intersection occurred (only if intersection occurred)
            normal: normal unit vector to plane of intersection
            Q0: first vertex of triangle intersected
            Q1: second vertex of triangle intersected
            Q2: third vertex of triangle intersected
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
        normal vector and a point on the plane, Q0. For debugging'''

        d = np.dot(normal, Q0)
        return np.abs(np.dot(normal,P0)-d)/np.linalg.norm(normal)



    def _calc_basic_stats(self, DIM3, t_indx=None):
        ''' Return basic stats about % agents remaining, fluid velocity, and 
        agent velocity for plot printing.
        DIM3 indicates the dimension of the domain, bool
        t_indx is the time index for pos_history or None for current time
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



    def plot(self, t=None, blocking=True, dist='density', fluid='vort', clip=None, figsize=None):
        '''Plot the position of the swarm at time t, or at the current time
        if no time is supplied. The actual time plotted will depend on the
        history of movement steps; the closest entry in
        environment.time_history will be shown without interpolation.
        
        Arguments:
            t: time to plot. if None, the current time.
            blocking: whether the plot should block execution or not
            dist: whether to plot Gaussian kernel density estimation or histogram.
                Options are:
                'density': plot Gaussian KDE using Scotts Factor from scipy.stats.gaussian_kde
                'cov': use the variance in each direction from self.shared_props['cov']
                float: a bandwidth factor to multiply the KDE variance by
                'hist': plot histogram
            fluid: fluid plot in background. 2D only!
                Options are:
                'vort': plot vorticity in the background
                'quiver': quiver plot of fluid velocity in the background
            clip: if plotting vorticity or FTLE, specifies the clip value for pseudocolor
            figsize: figure size. default is a heurstic that works... most of the time?
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
                flow = self.envir.interpolate_temporal_flow(t_indx=loc)
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
                x_density = stats.gaussian_kde(positions[:,0].compressed(), fac_x)
                y_density = stats.gaussian_kde(positions[:,1].compressed(), fac_y)
                axHistx.plot(xmesh, x_density(xmesh))
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
                x_density = stats.gaussian_kde(positions[:,0].compressed(), fac_x)
                y_density = stats.gaussian_kde(positions[:,1].compressed(), fac_y)
                z_density = stats.gaussian_kde(positions[:,2].compressed(), fac_z)
                axHistx.plot(xmesh, x_density(xmesh))
                axHisty.plot(ymesh, y_density(ymesh))
                axHistz.plot(zmesh, z_density(zmesh))
                axHistx.get_yaxis().set_ticks([])
                axHisty.get_yaxis().set_ticks([])
                axHistz.get_yaxis().set_ticks([])

        # show the plot
        plt.show(block=blocking)



    def plot_all(self, movie_filename=None, frames=None, downsamp=None, fps=10, 
                 dist='density', fluid=None, clip=None, figsize=None):
        ''' Plot the history of the swarm's movement, incl. current.
        If movie_filename is specified, output a movie file instead.
        
        Arguments:
            movie_filename: file name to save movie as
            frames: iterable of integers or None. If None, plot the entire
                history of the swarm's movement including the present. If an
                iterable, plot only the time steps of the swarm as indexed by
                the iterable.
            downsamp: iterable of integers, integer, or None. If None, do not
                downsample the agents. If an integer, plot only the first n 
                agents (equivalent to range(downsamp)). If an iterable, plot 
                only the agents specified. In all cases, statistics are reported
                for the TOTAL population, both shown and unshown. This includes
                the histograms.
            fps: frames per second, only used if saving a movie to file. Make
                sure this is at least a big as 1/dt, where dt is the time interval
                between frames!
            dist: whether to plot Gaussian kernel density estimation or histogram.
                Options are:
                'density': plot Gaussian KDE using Scotts Factor from scipy.stats.gaussian_kde
                'cov': use the variance in each direction from self.shared_props['cov']
                float: a bandwidth factor to multiply the KDE variance by
                'hist': plot histogram
            fluid: fluid plot in background. 2D only!
                Options are:
                'vort': plot vorticity in the background
                'quiver': quiver plot of fluid velocity in the background
            clip: if plotting vorticity or FTLE, specifies the clip value for pseudocolor
            figsize: figure size. default is a heurstic that works... most of the time?     
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
                        fac_x = None
                        fac_y = None
                xmesh = np.linspace(0, self.envir.L[0])
                ymesh = np.linspace(0, self.envir.L[1])
                x_density = stats.gaussian_kde(self.pos_history[n0][:,0].compressed(), fac_x)
                y_density = stats.gaussian_kde(self.pos_history[n0][:,1].compressed(), fac_y)
                xdens_plt, = axHistx.plot(xmesh, x_density(xmesh))
                ydens_plt, = axHisty.plot(y_density(ymesh),ymesh)
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
                x_density = stats.gaussian_kde(self.pos_history[n0][:,0].compressed(), fac_x)
                y_density = stats.gaussian_kde(self.pos_history[n0][:,1].compressed(), fac_y)
                z_density = stats.gaussian_kde(self.pos_history[n0][:,2].compressed(), fac_z)
                xdens_plt, = axHistx.plot(xmesh, x_density(xmesh))
                ydens_plt, = axHisty.plot(ymesh, y_density(ymesh))
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
                        x_density = stats.gaussian_kde(self.pos_history[n][:,0].compressed(), fac_x)
                        y_density = stats.gaussian_kde(self.pos_history[n][:,1].compressed(), fac_y)
                        xdens_plt.set_ydata(x_density(xmesh))
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
                        x_density = stats.gaussian_kde(self.pos_history[n][:,0].compressed(), fac_x)
                        y_density = stats.gaussian_kde(self.pos_history[n][:,1].compressed(), fac_y)
                        z_density = stats.gaussian_kde(self.pos_history[n][:,2].compressed(), fac_z)
                        xdens_plt.set_ydata(x_density(xmesh))
                        ydens_plt.set_ydata(y_density(ymesh))
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
                        x_density = stats.gaussian_kde(self.positions[:,0].compressed(), fac_x)
                        y_density = stats.gaussian_kde(self.positions[:,1].compressed(), fac_y)
                        xdens_plt.set_ydata(x_density(xmesh))
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
                        x_density = stats.gaussian_kde(self.positions[:,0].compressed(), fac_x)
                        y_density = stats.gaussian_kde(self.positions[:,1].compressed(), fac_y)
                        z_density = stats.gaussian_kde(self.positions[:,2].compressed(), fac_z)
                        xdens_plt.set_ydata(x_density(xmesh))
                        ydens_plt.set_ydata(y_density(ymesh))
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


    