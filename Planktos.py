#! /usr/bin/env python3

'''
Swarm class file, for simulating many individuals at once.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

import sys, warnings, pickle
from sys import platform
from pathlib import Path
from math import exp, log
import numpy as np
import numpy.ma as ma
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import animation
import mv_swarm
try:
    import data_IO
except ModuleNotFoundError:
    print("vtk libraries not loaded")

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class environment:

    def __init__(self, Lx=10, Ly=10, Lz=None,
                 x_bndry='zero', y_bndry='zero', z_bndry='noflux', flow=None,
                 flow_times=None, rho=None, mu=None, init_swarms=None, units='mm'):
        ''' Initialize environmental variables.

        Arguments:
            Lx: Length of domain in x direction, m
            Ly: Length of domain in y direction, m
            Lz: Length of domain in z direction, or None
            x_bndry: [left bndry condition, right bndry condition] (default: zero)
            y_bndry: [low bndry condition, high bndry condition] (default: zero)
            z_bndry: [low bndry condition, high bndry condition] (default: noflux)
            flow: [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k])]
                Note! i is x index, j is y index, with the value of x/y increasing
                as the index increases. It is assumed that the flow mesh is equally 
                spaced and includes values on the domain bndry
            flow_times: [tstart, tend] or iterable of times at which flow is specified
                     or scalar dt; required if flow is time-dependent.
            rho: fluid density of environment, kg/m**3 (optional, m here meaning length units)
            mu: dynamic viscosity, kg/m/s, Pa*s, N*s/m**2 (optional, m here meaning length units) 
            init_swarms: initial swarms in this environment
            units: length units to use

        Other properties:
            flow_points:
            fluid_domain_LLC:
            tiling:
            orig_L:
            a: height of porous region
            g: accel due to gravity (length units/s**2)
            struct_plots: List of functions that plot additional environment structures
            struct_plot_args: List of argument tuples to be passed to these functions, after ax
            time:
            time_history:

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

        ##### save flow #####
        self.flow_times = None
        self.flow_points = None
        self.fluid_domain_LLC = None
        self.tiling = None # (x,y) tiling amount
        self.orig_L = None # (Lx,Ly) before tiling/extending
        self.flow = flow

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

        # Fluid density kg/(length units)**3
        self.rho = rho
        # Dynamic viscosity kg/length units/s
        if mu == 0:
            raise RuntimeError("Dynamic viscosity, mu, cannot be zero.")
        else:
            self.mu = mu
        # porous region height
        self.a = None
        # accel due to gravity
        self.g = 9.80665

        ##### Environment Structure Plotting #####

        # List of functions that plot additional environment structures
        self.struct_plots = []
        # List of arguments tuples to be passed to these functions, after ax
        self.struct_plot_args = []

        ##### Initalize Time #####
        # By default, time is updated whenever an individual swarm moves (swarm.move()),
        #   or when all swarms in the environment are collectively moved.
        self.time = 0.0
        self.time_history = []



    def set_brinkman_flow(self, alpha, a, res, U, dpdx, tspan=None):
        '''Specify fully developed 2D or 3D flow with a porous region.
        Velocity gradient is zero in the x-direction; all flow moves parallel to
        the x-axis. Porous region is the lower part of the y-domain (2D) or
        z-domain (3D) with width=a and an empty region above. For 3D flow, the
        velocity profile is the same on all slices y=c. The decision to set
        2D vs. 3D flow is based on the dimension of the current domain.

        Arguments:
            alpha: equal to 1/(hydraulic permeability). alpha=0 implies free flow (infinitely permeable)
            a: height of porous region
            res: number of points at which to resolve the flow (int), including boundaries
            U: velocity at top of domain (v in input3d). scalar or list-like.
            dpdx: dp/dx change in momentum constant. scalar or list-like.
            tspan: [tstart, tend] or iterable of times at which flow is specified
                if None and U/dpdx are iterable, dt=1 will be used.

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.a = a

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
            b = self.L[1] - a
        else:
            y_mesh = np.linspace(0, self.L[2], res)
            b = self.L[2] - a

        ##### Calculate flow velocity #####
        flow = np.zeros((len(U), res, res)) # t, y, x
        t = 0
        for v, px in zip(U, dpdx):
            # Check for v = 0
            if v != 0:
                # Calculate C and D constants and then get A and B based on these

                D = ((exp(-alpha*a)/(alpha**2*self.mu)*px - exp(-2*alpha*a)*(
                    v/(1+alpha*b)+(2-alpha**2*b**2)*px/(2*alpha**2*self.mu*(1+alpha*b))))/
                    (1-exp(-2*alpha*a)*((1-alpha*b)/(1+alpha*b))))

                C = (v/(1+alpha*b) + (2-alpha**2*b**2)*px/((2*alpha**2*self.mu)*(1+alpha*b)) - 
                    D*(1-alpha*b)/(1+alpha*b))

                A = alpha*C - alpha*D
                B = C + D - px/(alpha**2*self.mu)

                for n, z in enumerate(y_mesh-a):
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
        self.a = a



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



    def set_two_layer_channel_flow(self, res, a, h_p, Cd, S):
        '''Apply wide-channel flow with vegetation layer according to the
        two-layer model described in Defina and Bixio (2005), 
        "Vegetated Open Channel Flow". The decision to set 2D vs. 3D flow is 
        based on the dimension of the domain and this flow is always
        time-independent by nature. The following parameters must be given:

        Arguments:
            res: number of points at which to resolve the flow (int), including boundaries
            a: vegetation density, given by Az*m, where Az is the frontal area
                of vegetation per unit depth and m the number of stems per unit area (1/m)
                (assumed constant)
            h_p: plant height (m)
            Cd: drag coefficient (assumed uniform) (unitless)
            S: bottom slope (unitless, 0-1 with 0 being no slope, resulting in no flow)

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.a = a

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
        self.a = h_p



    def set_canopy_flow(self, res, h, a, u_star=None, U_h=None, beta=0.3, C=0.25,
                        tspan=None):
        '''Apply flow within and above a uniform homogenous canopy according to the
        model described in Finnigan and Belcher (2004), 
        "Flow over a hill covered with a plant canopy". The decision to set 2D vs. 3D flow is 
        based on the dimension of the domain. Default values for beta and C are
        based on Finnigan & Belcher. Must specify two of u_star, U_h, and beta, 
        though beta has a default value of 0.3 so just giving u_star or U_h will also work. 
        If one of u_star, U_h, or beta is given as a list-like object, the flow will vary in time.

        Arguments:
            res: number of points at which to resolve the flow (int), including boundaries
            h: height of canopy (m)
            a: leaf area per unit volume of space m^{-1}. Typical values are
                a=1.0 for dense spruce to a=0.1 for open woodland
            u_star: canopy friction velocity. u_star = U_h*beta OR
            U_h: wind speed at top of canopy. U_h = u_star/beta
            beta: mass flux through the canopy (u_star/U_h)
            C: drag coefficient of indivudal canopy elements
            tspan: [tstart, tend] or iterable of times at which flow is specified
                if None and u_star, U_h, and/or beta are iterable, dt=1 will be used.

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.a = h

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
        self.a = h



    def read_IB2d_vtk_data(self, pathViz, dt, print_dump,
                           d_start=0, d_finish=None):
        '''Reads in vtk flow data generated by IB2d and sets environment
        variables accordingly.

        IB2d is an Immersed Boundary (IB) code for solving fully coupled
        fluid-structure interaction models in Python and MATLAB. The code is 
        hosted at https://github.com/nickabattista/IB2d

        Arguments:
            - pathViz: full path to viz_IB2d folder
            - dt: dt in input2d
            - print_dump: print_dump in input2d
            - d_start: number of first vtk dump to read in
            - d_finish: number of last vtk dump to read in, or None to read to end
        '''

        ##### Parse parameters and read in data #####

        pathViz = Path(pathViz)
        if pathViz.name != 'viz_IB2d':
            pathViz /= 'viz_IB2d'

        if d_finish is None:
            #infer d_finish
            file_names = [x.name for x in pathViz.iterdir() if x.is_file()]
            uX_nums = sorted([int(f[3:7]) for f in file_names if f[:3] == 'uX.'])
            d_finish = uX_nums[-1]

        X_vel = []
        Y_vel = []
        pathViz = str(pathViz) # Get string for passing into functions

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
            #    U: x-directed velocity   V: y-directed velocity
            #    Fx: x-directed Force     Fy: y-directed Force
            #
            #  Note: U(j,i): j-corresponds to y-index, i to the x-index
            #
            # read in x-directed Velocity Magnitude #
            strChoice = 'uX'; xy = True
            uX,x,y = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
                                                           strChoice,xy)
            X_vel.append(uX.T) # (y,x) -> (x,y) coordinates

            # read in y-directed Velocity Magnitude #
            strChoice = 'uY'
            uY = data_IO.read_2DEulerian_Data_From_vtk(pathViz,numSim,
                                                       strChoice)
            Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates

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

        ### Save data ###
        self.flow = [np.array(X_vel).squeeze(), np.array(Y_vel).squeeze()] 
        if d_start != d_finish:
            self.flow_times = np.arange(d_start,d_finish+1)*print_dump*dt
        else:
            self.flow_times = None
        # shift domain to quadrant 1
        self.flow_points = (x-x[0], y-y[0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(2)]
        self.__reset_flow_variables(incl_rho_mu=True)
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
        assert path.is_file(), "File {} not found!".format(filename)

        data, mesh, time = data_IO.read_vtk_Rectilinear_Grid_Vector(filename)

        self.flow = list(data)
        self.flow_times = None

        # shift domain to quadrant 1
        self.flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                            mesh[2]-mesh[2][0])

        ### Convert environment dimensions and reset simulation time ###
        self.L = [self.flow_points[dim][-1] for dim in range(3)]
        self.__reset_flow_variables(incl_rho_mu=True)
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
        assert path.exists(), "Path {} not found!".format(str(path))
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
        self.__reset_flow_variables(incl_rho_mu=True)
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
        assert path.exists(), "Path {} not found!".format(str(path))
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
        self.__reset_flow_variables(incl_rho_mu=True)
        self.reset()



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
            
        # update environment dimensions and meshes
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



    def add_swarm(self, swarm_s=100, init='random', seed=None, **kwargs):
        ''' Adds a swarm into this environment.

        Arguments:
            swarm_s: swarm object or size of the swarm (int)
            init: Method for initalizing positions.
            seed: Seed for random number generator
            kwargs: keyword arguments to be passed to the method for
                initalizing positions
        '''

        if isinstance(swarm_s, swarm):
            swarm_s.envir = self
            # check if the dimesion matches, project if possible
            if swarm_s.positions.shape[1] != len(self.L):
                if swarm_s.positions.shape[1] > len(self.L):
                    swarm_s.positions = swarm_s.positions[:,:2]
                    swarm_s.velocity = swarm_s.velocity[:,:2]
                    swarm_s.acceleration = swarm_s.acceleration[:,:2]
                else:
                    raise RuntimeError("Swarm dimension smaller than environment dimension!")
            self.swarms.append(swarm_s)
        else:
            return swarm(swarm_s, self, init=init, seed=seed, **kwargs)
            


    def move_swarms(self, dt=1.0, params=None):
        '''Move all swarms in the environment'''

        for s in self.swarms:
            s.move(dt, params, update_time=False)

        # update time
        self.time_history.append(self.time)
        self.time += dt



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



    def reset(self, rm_swarms=False):
        '''Resets environment to time=0. Swarm history will be lost, and all
        swarms will maintain their last position. If rm_swarms=True, remove
        all swarms.'''

        self.time = 0.0
        self.time_history = []
        if rm_swarms:
            self.swarms = []
        else:
            for sw in self.swarms:
                sw.pos_history = []



    def __reset_flow_variables(self, incl_rho_mu=False):
        '''To be used when the fluid flow changes. Resets all the helper
        parameters.'''

        self.a = None
        self.fluid_domain_LLC = None
        self.tiling = None
        self.orig_L = None
        self.plot_structs = []
        self.plot_structs_args = []
        if incl_rho_mu:
            self.mu = None
            self.rho = None



class swarm:

    def __init__(self, swarm_size=100, envir=None, init='random', seed=None, 
                 char_L=None, phys=None, **kwargs):
        ''' Initalizes planktos swarm in an environment.

        Arguments:
            swarm_size: Size of the swarm (int)
            envir: environment for the swarm, defaults to the standard environment
            init: Method for initalizing positions.
            seed: Seed for random number generator, int or None
            char_L: characteristic length
            phys: dictionary of physical properties to be used by equations of motion
            kwargs: keyword arguments to be passed to the method for
                initalizing positions

        Methods for initializing the swarm:
            - 'random': Uniform random distribution throughout the domain
            - 'point': All positions set to a single point.
                Required keyword arguments:
                - x = (float) x-coordinate
                - y = (float) y-coordinate
                - z = (optional, float) z-coordinate, if 3D

        To customize agent behavior, subclass this class and re-implement the
        method update_positions (do not change the call signature.
        '''

        # use a new, 3D default environment if one was not given
        if envir is None:
            self.envir = environment(init_swarms=self, Lz=10)
        else:
            try:
                assert envir.__class__.__name__ == 'environment'
                envir.swarms.append(self)
                self.envir = envir
            except AssertionError:
                print("Error: invalid environment object.")
                raise

        # set physical properties
        self.char_L = char_L
        self.phys = phys

        # initialize random number generator
        self.rndState = np.random.RandomState(seed=seed)

        # initialize bug locations
        self.positions = ma.zeros((swarm_size, len(self.envir.L)))
        self.positions.harden_mask() # prevent unintentional mask overwites
        mv_swarm.init_pos(self, init, kwargs)

        # initialize bug velocities
        self.velocity = ma.zeros((swarm_size, len(self.envir.L)))
        self.velocity.harden_mask()

        # initialize bug accelerations
        self.acceleration = ma.zeros((swarm_size, len(self.envir.L)))
        self.acceleration.harden_mask()

        # Initialize position history
        self.pos_history = []

        # Apply boundary conditions in case of domain mismatch
        self.apply_boundary_conditions()



    def calc_re(self, u):
        '''Calculate Reynolds number based on environment variables and given
        flow velocity, u'''

        if self.envir.rho is not None and self.envir.mu is not None and\
            self.char_L is not None:
            return self.envir.rho*u*self.char_L/self.envir.mu
        else:
            raise RuntimeError("Parameters necessary for Re calculation are undefined.")



    def move(self, dt=1.0, params=None, update_time=True):
        '''Move all organisms in the swarm over an amount of time dt.
        Do not override this method when subclassing - override update_positions
        instead!

        Arguments:
            dt: time-step for move
            params: iterable of parameters to pass along to update_positions
            update_time: whether or not to update the environment's time by dt
        '''

        # Put current position in the history
        self.pos_history.append(self.positions.copy())

        # Check that something is left in the domain to move, and move it.
        if not np.all(self.positions.mask):
            # Update positions
            self.update_positions(dt, params)
            # Apply boundary conditions.
            self.apply_boundary_conditions()

        # Record new time
        if update_time:
            self.envir.time_history.append(self.envir.time)
            self.envir.time += dt

            # Check for other swarms in environment and freeze them
            warned = False
            for s in self.envir.swarms:
                if s is not self and len(s.pos_history) < len(self.pos_history):
                    s.pos_history.append(s.positions)
                    if not warned:
                        warnings.warn("Other swarms in the environment were not"+
                                      " moved during this environmental timestep.\n"+
                                      "If this was not your intent, call"+
                                      " envir.move_swarms instead of this method"+
                                      " to move all the swarms at once.",
                                      UserWarning)
                        warned = True



    def update_positions(self, dt, params):
        '''Update agent positions.
        THIS IS THE METHOD TO OVERRIDE IF YOU WANT DIFFERENT MOVEMENT!
        Note: do not change the call signature.
        The result of this method should be to overwrite self.positions with
        the new agent positions after a time step of length dt. It should also
        update the agent velocities at the end of the timestep in self.velocity.

        What is included in this implementation is a basic jitter behavior with
        drift according to the fluid flow.
        params = (mean=0, covariance matrix=np.eye)

        Arguments:
            dt: length of time step
            params: iterable of any other parameters necessary
        '''

        # 3D?
        DIM3 = (len(self.envir.L) == 3)

        # Parse optional parameters
        if params is not None:
            assert isinstance(params[1], np.ndarray), "cov must be ndarray"
            if not DIM3:
                assert len(params[0]) == 2, "mu must be length 2"
                assert params[1].shape == (2,2), "cov must be shape (2,2)"
            else:
                assert len(params[0]) == 3, "mu must be length 3"
                assert params[1].shape == (3,3), "cov must be shape (3,3)"
        else:
            params = (np.zeros(len(self.envir.L)), 0.01*np.eye(len(self.envir.L)))


        ### Passive movement ###
        # Get fluid-based drift and add to Gaussian bias
        mu = self.get_fluid_drift() + params[0]
        #mu = mv_swarm.massive_drift(self, dt) + params[0]

        ### Active movement ###
        # Add jitter and move according to a Gaussian random walk.
        mv_swarm.gaussian_walk(self, dt*mu, dt*params[1])

        # Update velocity of swarm
        self.velocity = (self.positions - self.pos_history[-1])/dt



    def get_fluid_drift(self):
        '''Return fluid-based drift for all agents via interpolation.
        
        In the returned 2D ndarray, each row corresponds to an agent (in the
        same order as listed in self.positions) and each column is a dimension.
        '''
        
        # 3D?
        DIM3 = (len(self.envir.L) == 3)

        # Interpolate fluid flow
        if self.envir.flow is None:
            if not DIM3:
                return np.array([0, 0])
            else:
                return np.array([0, 0, 0])
        else:
            if (not DIM3 and len(self.envir.flow[0].shape) == 2) or \
               (DIM3 and len(self.envir.flow[0].shape) == 3):
                # temporally constant flow
                return self.__interpolate_flow(self.envir.flow, method='linear')
            else:
                # temporal flow. interpolate in time, and then in space.
                return self.__interpolate_flow(self.__interpolate_temporal_flow(),
                                             method='linear')



    def get_projectile_motion(self, high_re=False):
        '''Return acceleration using equations of projectile motion.
        Includes drag, inertia, and background flow velocity. Does not include
        gravity.

        Arguments:
            high_re: If false (default), assume Re<0.1 for all agents. Otherwise,
            assume Re > 10 for all agents.
        
        Requires that the following are specified in self.phys:
            Cd: Drag coefficient
            S: cross-sectional area of each agent
            m: mass of each agent
            L: diameter of the agent (low Re only)

        Requires that the following are specified in envir:
            rho: fluid density
            mu: dynamic viscosity (if low Re)

        Returns:
            2D array of accelerations in each dimension for each agent.
            Each row corresponds to an agent (in the same order as listed in 
            self.positions) and each column is a dimension.
        '''
        
        # Get fluid velocity
        vel = self.get_fluid_drift()

        # Check for self.phys and other parameters
        assert isinstance(self.phys, dict), "swarm.phys not specified"
        for key in ['Cd', 'm']:
            assert key in self.phys, "{} not found in swarm.phys".format(key)
        assert self.envir.rho is not None, "rho not specified"
        if not high_re:
            assert self.envir.mu is not None, "mu not specified"
            assert self.char_L is not None, "characteristic length not specified"
        else:
            assert 'S' in self.phys, "Cross-sectional area (S) not found in "+\
            "swarm.phys"

        if high_re:
            diff = np.linalg.norm(self.velocity-vel,axis=1)
            return self.acceleration/self.phys['m'] -\
            (self.envir.rho*self.phys['Cd']*self.phys['S']/2/self.phys['m'])*\
            (self.velocity - vel)*np.stack((diff,diff,diff)).T
        else:
            return self.acceleration/self.phys['m'] -\
            (self.envir.mu*self.phys['Cd']*self.char_L/2/self.phys['m'])*\
            (self.velocity - vel)



    def apply_boundary_conditions(self):
        '''Apply boundary conditions to self.positions'''

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
                self.velocity[maskrow, :] = ma.masked
                self.acceleration[maskrow, :] = ma.masked
            elif bndry[0] == 'noflux':
                # pull everything exiting on the left to 0
                zerorow = self.positions[:,dim] < 0
                self.positions[zerorow, dim] = 0
                self.velocity[zerorow, dim] = 0
                self.acceleration[zerorow, dim] = 0
            else:
                raise NameError

            ### Right boundary ###
            if bndry[1] == 'zero':
                # mask everything exiting on the right
                maskrow = self.positions[:,dim] > self.envir.L[dim]
                self.positions[maskrow, :] = ma.masked
                self.velocity[maskrow, :] = ma.masked
                self.acceleration[maskrow, :] = ma.masked
            elif bndry[1] == 'noflux':
                # pull everything exiting on the left to 0
                zerorow = self.positions[:,dim] > self.envir.L[dim]
                self.positions[zerorow, dim] = self.envir.L[dim]
                self.velocity[zerorow, dim] = 0
                self.acceleration[zerorow, dim] = 0
            else:
                raise NameError



    def __interpolate_flow(self, flow, method):
        '''Interpolate the fluid velocity field at swarm positions'''

        x_vel = interpolate.interpn(self.envir.flow_points, flow[0],
                                    self.positions, method=method)
        y_vel = interpolate.interpn(self.envir.flow_points, flow[1],
                                    self.positions, method=method)
        if len(flow) == 3:
            z_vel = interpolate.interpn(self.envir.flow_points, flow[2],
                                        self.positions, method=method)
            return np.array([x_vel, y_vel, z_vel]).T
        else:
            return np.array([x_vel, y_vel]).T



    def __interpolate_temporal_flow(self, t_indx=None):
        '''Linearly interpolate flow in time
        t_indx is the time index for pos_history or None for current time
        '''

        if t_indx is None:
            time = self.envir.time
        else:
            time = self.envir.time_history[t_indx]

        # boundary cases
        if time <= self.envir.flow_times[0]:
            return [f[0, ...] for f in self.envir.flow]
        elif time >= self.envir.flow_times[-1]:
            return [f[-1, ...] for f in self.envir.flow]
        else:
            # linearly interpolate
            indx = np.searchsorted(self.envir.flow_times, time)
            diff = self.envir.flow_times[indx] - time
            dt = self.envir.flow_times[indx] - self.envir.flow_times[indx-1]
            return [f[indx, ...]*(dt-diff)/dt + f[indx-1, ...]*diff/dt
                    for f in self.envir.flow]



    def __plot_setup(self, fig):
        ''' Setup figures for plotting '''

        if len(self.envir.L) == 2:
            # 2D plot

            # chop up the axes to include histograms
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            bottom_h = left_h = left + width + 0.02

            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom_h, width, 0.2]
            rect_histy = [left_h, bottom, 0.2, height]

            ax = plt.axes(rect_scatter, xlim=(0, self.envir.L[0]), 
                          ylim=(0, self.envir.L[1]))
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

            # add a grassy porous layer background (if porous layer present)
            if self.envir.a is not None:
                grass = np.random.rand(80)*self.envir.L[0]
                for g in grass:
                    ax.axvline(x=g, ymax=self.envir.a/self.envir.L[1], color='.5')

            # plot any structures
            for plot_func, args in zip(self.envir.plot_structs, 
                                       self.envir.plot_structs_args):
                plot_func(ax, *args)

            return ax, axHistx, axHisty

        else:
            # 3D plot

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
            ax = Axes3D(fig, rect=rect_scatter) #fig.add_subplot(121, projection='3d')
            ax.set_xlim((0, self.envir.L[0]))
            ax.set_ylim((0, self.envir.L[1]))
            ax.set_zlim((0, self.envir.L[2]))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Organism positions')
            # No real solution to 3D aspect ratio...
            #ax.set_aspect('equal','box')

            # histograms
            int_ticks = MaxNLocator(nbins='auto', integer=True)
            axHistx = plt.axes(rect_histx)
            axHistx.set_xlim((0, self.envir.L[0]))
            axHistx.yaxis.set_major_locator(int_ticks)
            axHistx.set_ylabel('X    ', rotation=0)
            axHisty = plt.axes(rect_histy)
            axHisty.set_xlim((0, self.envir.L[1]))
            axHisty.yaxis.set_major_locator(int_ticks)
            axHisty.set_ylabel('Y    ', rotation=0)
            axHistz = plt.axes(rect_histz)
            axHistz.set_xlim((0, self.envir.L[2]))
            axHistz.yaxis.set_major_locator(int_ticks)
            axHistz.set_ylabel('Z    ', rotation=0)

            # add a grassy porous layer background (if porous layer present)
            if self.envir.a is not None:
                grass = np.random.rand(120,2)
                grass[:,0] *= self.envir.L[0]
                grass[:,1] *= self.envir.L[1]
                for g in grass:
                    ax.plot([g[0],g[0]], [g[1],g[1]], [0,self.envir.a],
                            'k-', alpha=0.5)

            # plot any structures
            for plot_func, args in zip(self.envir.plot_structs, 
                                       self.envir.plot_structs_args):
                plot_func(ax, *args)

            return ax, axHistx, axHisty, axHistz



    def __calc_basic_stats(self, DIM3, t_indx=None):
        ''' Return basic stats about % remaining and flow for plot printing.
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

        if not DIM3:
            # 2D flow
            # get current fluid flow info
            if len(self.envir.flow[0].shape) == 2:
                # temporally constant flow
                flow = self.envir.flow
            else:
                # temporally changing flow
                flow = self.__interpolate_temporal_flow(t_indx)
            flow_spd = np.sqrt(flow[0]**2 + flow[1]**2)
            avg_spd_x = flow[0].mean()
            avg_spd_y = flow[1].mean()
            avg_spd = flow_spd.mean()
            max_spd = flow_spd.max()
            return perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y

        else:
            # 3D flow
            if len(self.envir.flow[0].shape) == 3:
                # temporally constant flow
                flow = self.envir.flow
            else:
                # temporally changing flow
                flow = self.__interpolate_temporal_flow(t_indx)
            flow_spd = np.sqrt(flow[0]**2 + flow[1]**2 + flow[2]**2)
            avg_spd_x = flow[0].mean()
            avg_spd_y = flow[1].mean()
            avg_spd_z = flow[2].mean()
            avg_spd = flow_spd.mean()
            max_spd = flow_spd.max()
            return perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z



    def plot(self, t=None, blocking=True):
        '''Plot the position of the swarm at time t, or at the current time
        if no time is supplied. The actual time plotted will depend on the
        history of movement steps; the closest entry in
        environment.time_history will be shown without interpolation.'''

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
            aspectratio = self.envir.L[0]/self.envir.L[1]
            fig = plt.figure(figsize=(5*aspectratio+1,6))
            ax, axHistx, axHisty = self.__plot_setup(fig)

            # scatter plot and time text
            ax.scatter(positions[:,0], positions[:,1], label='organism')
            ax.text(0.02, 0.95, 'time = {:.2f}'.format(time),
                    transform=ax.transAxes, fontsize=12)

            # textual info
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y = \
                self.__calc_basic_stats(DIM3=False, t_indx=loc)
            plt.figtext(0.77, 0.81,
                        '{:.1f}% remain\n'.format(perc_left)+
                        '\n------Flow info------\n'+
                        'Avg vel: {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                        'Max vel: {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                        'Avg x vel: {:.1g} {}/s\n'.format(avg_spd_x, self.envir.units)+
                        'Avg y vel: {:.1g} {}/s'.format(avg_spd_y, self.envir.units),
                        fontsize=10)

            # histograms
            bins_x = np.linspace(0, self.envir.L[0], 26)
            bins_y = np.linspace(0, self.envir.L[1], 26)
            axHistx.hist(positions[:,0].compressed(), bins=bins_x)
            axHisty.hist(positions[:,1].compressed(), bins=bins_y,
                         orientation='horizontal')

        else:
            # 3D plot
            fig = plt.figure(figsize=(10,5))
            ax, axHistx, axHisty, axHistz = self.__plot_setup(fig)

            # scatter plot and time text
            ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                       label='organism')
            ax.text2D(0.02, 1, 'time = {:.2f}'.format(time),
                      transform=ax.transAxes, verticalalignment='top',
                      fontsize=12)

            # textual info
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z = \
                self.__calc_basic_stats(DIM3=True, t_indx=loc)
            ax.text2D(1, 0.945, 'Avg vel: {:.2g} {}/s\n'.format(avg_spd, self.envir.units)+
                      'Max vel: {:.2g} {}/s'.format(max_spd, self.envir.units),
                      transform=ax.transAxes, horizontalalignment='right',
                      fontsize=10)
            ax.text2D(0.02, 0, '{:.1f}% remain\n'.format(perc_left),
                      transform=fig.transFigure, fontsize=10)
            axHistx.text(0.02, 0.98, 'Avg x vel: {:.2g} {}/s\n'.format(avg_spd_x,
                         self.envir.units),
                         transform=axHistx.transAxes, verticalalignment='top',
                         fontsize=10)
            axHisty.text(0.02, 0.98, 'Avg y vel: {:.2g} {}/s\n'.format(avg_spd_y,
                         self.envir.units),
                         transform=axHisty.transAxes, verticalalignment='top',
                         fontsize=10)
            axHistz.text(0.02, 0.98, 'Avg z vel: {:.2g} {}/s\n'.format(avg_spd_z,
                         self.envir.units),
                         transform=axHistz.transAxes, verticalalignment='top',
                         fontsize=10)

            # histograms
            bins_x = np.linspace(0, self.envir.L[0], 26)
            bins_y = np.linspace(0, self.envir.L[1], 26)
            bins_z = np.linspace(0, self.envir.L[2], 26)
            axHistx.hist(positions[:,0].compressed(), bins=bins_x, alpha=0.8)
            axHisty.hist(positions[:,1].compressed(), bins=bins_y, alpha=0.8)
            axHistz.hist(positions[:,2].compressed(), bins=bins_z, alpha=0.8)

        # show the plot
        plt.show(blocking)



    def plot_all(self, movie_filename=None, fps=10):
        ''' Plot the entire history of the swarm's movement, incl. current.
        If movie_filename is specified, output a movie file instead.'''

        if len(self.envir.time_history) == 0:
            print('No position history! Plotting current position...')
            self.plot()
            return
        
        DIM3 = (len(self.envir.L) == 3)

        if not DIM3:
            ### 2D setup ###
            aspectratio = self.envir.L[0]/self.envir.L[1]
            fig = plt.figure(figsize=(5*aspectratio+1,6))
            ax, axHistx, axHisty = self.__plot_setup(fig)

            scat = ax.scatter([], [], label='organism')

            # textual info
            time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                                fontsize=12)
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y = \
                self.__calc_basic_stats(DIM3=False, t_indx=0)
            axStats = plt.axes([0.77, 0.77, 0.25, 0.2], frameon=False)
            axStats.set_axis_off()
            stats_text = axStats.text(0,1,
                         '{:.1f}% remain\n'.format(perc_left)+
                         '\n------Flow info------\n'+
                         'Avg vel: {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                         'Max vel: {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                         'Avg x vel: {:.1g} {}/s\n'.format(avg_spd_x, self.envir.units)+
                         'Avg y vel: {:.1g} {}/s'.format(avg_spd_y, self.envir.units),
                         fontsize=10, transform=axStats.transAxes,
                         verticalalignment='top')

            # histograms
            data_x = self.pos_history[0][:,0].compressed()
            data_y = self.pos_history[0][:,1].compressed()
            bins_x = np.linspace(0, self.envir.L[0], 26)
            bins_y = np.linspace(0, self.envir.L[1], 26)
            n_x, bins_x, patches_x = axHistx.hist(data_x, bins=bins_x)
            n_y, bins_y, patches_y = axHisty.hist(data_y, bins=bins_y, 
                                                  orientation='horizontal')
            
        else:
            ### 3D setup ###
            fig = plt.figure(figsize=(10,5))
            ax, axHistx, axHisty, axHistz = self.__plot_setup(fig)

            scat = ax.scatter(self.pos_history[0][:,0], self.pos_history[0][:,1],
                              self.pos_history[0][:,2], label='organism',
                              animated=True)

            # textual info
            time_text = ax.text2D(0.02, 1, 'time = {:.2f}'.format(
                                  self.envir.time_history[0]),
                                  transform=ax.transAxes, animated=True,
                                  verticalalignment='top', fontsize=12)
            perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z = \
                self.__calc_basic_stats(DIM3=True, t_indx=0)
            flow_text = ax.text2D(1, 0.945,
                                  'Avg vel: {:.2g} {}/s\n'.format(avg_spd, self.envir.units)+
                                  'Max vel: {:.2g} {}/s'.format(max_spd, self.envir.units),
                                  transform=ax.transAxes, animated=True,
                                  horizontalalignment='right', fontsize=10)
            perc_text = ax.text2D(0.02, 0,
                                  '{:.1f}% remain\n'.format(perc_left),
                                  transform=fig.transFigure, animated=True,
                                  fontsize=10)
            x_flow_text = axHistx.text(0.02, 0.98,
                                       'Avg x vel: {:.2g} {}/s\n'.format(avg_spd_x,
                                       self.envir.units),
                                       transform=axHistx.transAxes, animated=True,
                                       verticalalignment='top', fontsize=10)
            y_flow_text = axHisty.text(0.02, 0.98,
                                       'Avg y vel: {:.2g} {}/s\n'.format(avg_spd_y,
                                       self.envir.units),
                                       transform=axHisty.transAxes, animated=True,
                                       verticalalignment='top', fontsize=10)
            z_flow_text = axHistz.text(0.02, 0.98,
                                       'Avg z vel: {:.2g} {}/s\n'.format(avg_spd_z,
                                       self.envir.units),
                                       transform=axHistz.transAxes, animated=True,
                                       verticalalignment='top', fontsize=10)

            # histograms
            data_x = self.pos_history[0][:,0].compressed()
            data_y = self.pos_history[0][:,1].compressed()
            data_z = self.pos_history[0][:,2].compressed()
            bins_x = np.linspace(0, self.envir.L[0], 26)
            bins_y = np.linspace(0, self.envir.L[1], 26)
            bins_z = np.linspace(0, self.envir.L[2], 26)
            n_x, bins_x, patches_x = axHistx.hist(data_x, bins=bins_x, alpha=0.8)
            n_y, bins_y, patches_y = axHisty.hist(data_y, bins=bins_y, alpha=0.8)
            n_z, bins_z, patches_z = axHistz.hist(data_z, bins=bins_z, alpha=0.8)

        # animation function. Called sequentially
        def animate(n):
            if n < len(self.pos_history):
                time_text.set_text('time = {:.2f}'.format(self.envir.time_history[n]))
                if not DIM3:
                    # 2D
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y = \
                        self.__calc_basic_stats(DIM3=False, t_indx=n)
                    stats_text.set_text('{:.1f}% remain\n'.format(perc_left)+
                                        '\n------Flow info------\n'+
                                        'Avg vel: {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                                        'Max vel: {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                                        'Avg x vel: {:.1g} {}/s\n'.format(avg_spd_x, self.envir.units)+
                                        'Avg y vel: {:.1g} {}/s'.format(avg_spd_y, self.envir.units))
                    scat.set_offsets(self.pos_history[n])
                    n_x, _ = np.histogram(self.pos_history[n][:,0].compressed(), bins_x)
                    n_y, _ = np.histogram(self.pos_history[n][:,1].compressed(), bins_y)
                    for rect, h in zip(patches_x, n_x):
                        rect.set_height(h)
                    for rect, h in zip(patches_y, n_y):
                        rect.set_width(h)
                    return [scat, time_text, stats_text]+patches_x+patches_y
                    
                else:
                    # 3D
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z = \
                        self.__calc_basic_stats(DIM3=True, t_indx=n)
                    flow_text.set_text('Avg vel: {:.2g} {}/s\n'.format(avg_spd, self.envir.units)+
                                       'Max vel: {:.2g} {}/s'.format(max_spd, self.envir.units))
                    perc_text.set_text('{:.1f}% remain\n'.format(perc_left))
                    x_flow_text.set_text('Avg x vel: {:.2g} {}/s\n'.format(avg_spd_x, self.envir.units))
                    y_flow_text.set_text('Avg y vel: {:.2g} {}/s\n'.format(avg_spd_y, self.envir.units))
                    z_flow_text.set_text('Avg z vel: {:.2g} {}/s\n'.format(avg_spd_z, self.envir.units))
                    scat._offsets3d = (np.ma.ravel(self.pos_history[n][:,0].compressed()),
                                       np.ma.ravel(self.pos_history[n][:,1].compressed()),
                                       np.ma.ravel(self.pos_history[n][:,2].compressed()))
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
                    return [scat, time_text, flow_text, perc_text, x_flow_text, 
                        y_flow_text, z_flow_text]+patches_x+patches_y+patches_z
                    
            else:
                time_text.set_text('time = {:.2f}'.format(self.envir.time))
                if not DIM3:
                    # 2D end
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y = \
                        self.__calc_basic_stats(DIM3=False, t_indx=None)
                    stats_text.set_text('{:.1f}% remain\n'.format(perc_left)+
                                        '\n-----Flow info-----\n'+
                                        'Avg vel: {:.1g} {}/s\n'.format(avg_spd, self.envir.units)+
                                        'Max vel: {:.1g} {}/s\n'.format(max_spd, self.envir.units)+
                                        'Avg x vel: {:.1g} {}/s\n'.format(avg_spd_x, self.envir.units)+
                                        'Avg y vel: {:.1g} {}/s'.format(avg_spd_y, self.envir.units))
                    scat.set_offsets(self.positions)
                    n_x, _ = np.histogram(self.positions[:,0].compressed(), bins_x)
                    n_y, _ = np.histogram(self.positions[:,1].compressed(), bins_y)
                    for rect, h in zip(patches_x, n_x):
                        rect.set_height(h)
                    for rect, h in zip(patches_y, n_y):
                        rect.set_width(h)
                    return [scat, time_text, stats_text]+patches_x+patches_y
                    
                else:
                    # 3D end
                    perc_left, avg_spd, max_spd, avg_spd_x, avg_spd_y, avg_spd_z = \
                        self.__calc_basic_stats(DIM3=True)
                    flow_text.set_text('Avg vel: {:.2g} {}/s\n'.format(avg_spd, self.envir.units)+
                                       'Max vel: {:.2g} {}/s'.format(max_spd, self.envir.units))
                    perc_text.set_text('{:.1f}% remain\n'.format(perc_left))
                    x_flow_text.set_text('Avg x vel: {:.2g} {}/s\n'.format(avg_spd_x, self.envir.units))
                    y_flow_text.set_text('Avg y vel: {:.2g} {}/s\n'.format(avg_spd_y, self.envir.units))
                    z_flow_text.set_text('Avg z vel: {:.2g} {}/s\n'.format(avg_spd_z, self.envir.units))
                    scat._offsets3d = (np.ma.ravel(self.positions[:,0].compressed()),
                                       np.ma.ravel(self.positions[:,1].compressed()),
                                       np.ma.ravel(self.positions[:,2].compressed()))
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
                    return [scat, time_text, flow_text, perc_text, x_flow_text, 
                        y_flow_text, z_flow_text]+patches_x+patches_y+patches_z

        # infer animation rate from dt between current and last position
        dt = self.envir.time - self.envir.time_history[-1]

        anim = animation.FuncAnimation(fig, animate, frames=len(self.pos_history)+1,
                                    interval=dt*100, repeat=False, blit=True,
                                    save_count=len(self.pos_history)+1)

        if movie_filename is not None:
            try:
                writer = animation.FFMpegWriter(fps=fps, 
                    metadata=dict(artist='Christopher Strickland'))#, bitrate=1800)
                anim.save(movie_filename, writer=writer, dpi=300)
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
