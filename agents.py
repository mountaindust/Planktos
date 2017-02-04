#! /usr/bin/env python3

'''
Swarm class file, for simulating many individuals at once.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import sys
from math import exp, log
import numpy as np
import numpy.ma as ma
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import animation
import init_pos

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class environment:

    def __init__(self, Lx=100, Ly=100, x_bndry=None, y_bndry=None, flow=None,
                 flow_times=None, Re=None, rho=None, init_swarms=None):
        ''' Initialize environmental variables.

        Arguments:
            Lx: Length of domain in x direction
            Ly: Length of domain in y direction
            x_bndry: [left bndry condition, right bndry condition]
            y_bndry: [low bndry condition, high bndry condition]
            flow: [x-vel field ndarray ([t],m,n), y-vel field ndarray ([t],m,n)]
                Note! y=0 is at row zero, increasing downward, and it is assumed
                that flow mesh is equally spaced incl values on the domain bndry
            flow_times: [tstart, tend] or iterable of times at which flow is specified
                     or scalar dt; required if flow is time-dependent.
            Re: Reynolds number of environment (optional)
            rho: fluid density of environment, kh/m**3 (optional)
            init_swarms: initial swarms in this environment

        Right now, supported boundary conditions are 'zero' (default) and 'noflux'.
        '''

        # Save domain size
        self.L = [Lx, Ly]

        # Parse boundary conditions
        supprted_conds = ['zero', 'noflux']
        self.bndry = []

        if x_bndry is None:
            # default boundary conditions
            self.bndry.append(['zero', 'zero'])
        elif x_bndry[0] not in supprted_conds or x_bndry[1] not in supprted_conds:
            raise NameError("X boundary condition {} not implemented.".format(x_bndry))
        else:
            self.bndry.append(x_bndry)
        if y_bndry is None:
            # default boundary conditions
            self.bndry.append(['zero', 'zero'])
        elif y_bndry[0] not in supprted_conds or y_bndry[1] not in supprted_conds:
            raise NameError("Y boundary condition {} not implemented.".format(y_bndry))
        else:
            self.bndry.append(y_bndry)

        # save flow
        self.flow_times = None
        self.flow_points = None
        if flow is not None:
            try:
                assert isinstance(flow, list)
                assert len(flow) > 1
                for ii in range(len(flow)):
                    assert isinstance(flow[ii], np.ndarray)
            except AssertionError:
                tb = sys.exc_info()[2]
                raise AttributeError(
                    'flow must be specified as a list of ndarrays.').with_traceback(tb)
            if max([len(f.shape) for f in flow]) > 2:
                # time-dependent flow
                assert flow[0].shape[0] == flow[1].shape[0]
                assert flow_times is not None
                self.__set_flow_variables(flow_times)
            else:
                self.__set_flow_variables()
        self.flow = flow

        # swarm list
        if init_swarms is None:
            self.swarms = []
        else:
            if isinstance(init_swarms, list):
                self.swarms = init_swarms
            else:
                self.swarms = [init_swarms]
            # reset each swarm's environment
            for swarm in self.swarms:
                swarm.envir = self

        ##### Fluid Variables #####

        # Re
        self.re = Re
        # Fluid density kg/m**3
        self.rho = rho
        # Characteristic length
        self.char_L = self.L[1]
        # porous region height
        self.a = None

        ##### Initalize Time #####
        # By default, time is updated whenever an individual swarm moves (swarm.move()),
        #   or when all swarms in the environment are collectively moved.
        self.time = 0.0
        self.time_history = []



    def set_brinkman_flow(self, alpha, a, res, U, dpdx, tspan=None):
        '''Specify fully developed 2D flow with a porous region.
        Velocity gradient is zero in the x-direction; porous region is the lower
        part of the y-domain (width=a) with an empty region above.

        Arguments:
            alpha: porosity constant
            a: height of porous region
            res: number of points at which to resolve the flow (int)
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

        # Parse parameters
        if hasattr(U, '__iter__'):
            try:
                assert hasattr(dpdx, '__iter__')
                assert len(dpdx) == len(U)
            except AssertionError:
                print('dpdx must be the same length as U.')
                raise
        else:
            try:
                assert not hasattr(dpdx, '__iter__')
            except AssertionError:
                print('dpdx must be the same length as U.')
                raise
            U = [U]
            dpdx = [dpdx]

        for v in U:
            assert v != 0, "U cannot be equal to zero."

        if self.re is None or self.rho is None:
            print('Fluid properties of environment are unspecified.')
            print('Re = {}'.format(self.re))
            print('Rho = {}'.format(self.rho))
            raise AttributeError

        b = self.L[1] - a
        self.a = a

        # Get y-mesh
        y_mesh = np.linspace(0, self.L[1], res)

        # Calculate flow velocity
        flow = np.zeros((len(U), res, res))
        t = 0
        for v, px in zip(U, dpdx):
            mu = self.rho*v*self.char_L/self.re

            # Calculate C and D constants and then get A and B based on these

            C = (px*(-0.5*alpha**2*b**2+exp(log(alpha*b)-alpha*a)-exp(-alpha*a)+1) +
                v*alpha**2*mu)/(alpha**2*mu*(exp(log(alpha*b)-2*alpha*a)+alpha*b-
                exp(-2*alpha*a)+1))

            D = (px*(exp(log(0.5*alpha**2*b**2)-2*alpha*a)+exp(log(alpha*b)-alpha*a)+
                exp(-alpha*a)-exp(-2*alpha*a)) - 
                exp(log(v*alpha**2*mu)-2*alpha*a))/(alpha**2*mu*
                (exp(log(alpha*b)-2*alpha*a)+alpha*b-exp(-2*alpha*a)+1))

            A = alpha*C - alpha*D
            B = C + D - px/(alpha**2*mu)

            for n, z in enumerate(y_mesh-a):
                if z > 0:
                    #Region I
                    flow[t,n,:] = z**2*px/(2*mu) + A*z + B
                else:
                    #Region 2
                    if C > 0 and D > 0:
                        flow[t,n,:] = exp(log(C)+alpha*z) + exp(log(D)-alpha*z) - px/(alpha**2*mu)
                    elif C <= 0 and D > 0:
                        flow[t,n,:] = exp(log(D)-alpha*z) - px/(alpha**2*mu)
                    elif C > 0 and D <= 0:
                        flow[t,n,:] = exp(log(C)+alpha*z) - px/(alpha**2*mu)
                    else:
                        flow[t,n,:] = -px/(alpha**2*mu)
            t += 1

        flow = flow.squeeze()
        self.flow = [flow, np.zeros_like(flow)] #x-direction, y-direction

        if len(U) == 1:
            self.__set_flow_variables()
        else:
            if tspan is None:
                self.__set_flow_variables(tspan=1)
            else:
                self.__set_flow_variables(tspan=tspan)
                



    def add_swarm(self, swarm_s=100, init='random', **kwargs):
        ''' Adds a swarm into this environment.

        Arguments:
            swarm_s: swarm object or size of the swarm (int)
            init: Method for initalizing positions.
            kwargs: keyword arguments to be passed to the method for
                initalizing positions
        '''

        if isinstance(swarm_s, swarm):
            swarm_s.envir = self
            self.swarms.append(swarm_s)
        else:
            swarm(swarm_s, self, init, **kwargs)



    def move_swarms(self, dt=1.0, params=None):
        '''Move all swarms in the environment'''

        for s in self.swarms:
            s.move(dt, params, update_time=False)

        # update time
        self.time_history.append(self.time)
        self.time += dt



    def __set_flow_variables(self, tspan=None):
        '''Store points at which flow is specified, and time information. 
        
        Arguments:
            tspan: [tstart, tend] or iterable of times at which flow is specified
                    or scalar dt. Required if flow is time-dependent; None will 
                    be interpreted as non time-dependent flow.
        '''
        if tspan is None:
            # no time-dependent flow
            x_f_mesh = np.linspace(0,self.L[0],self.flow[0].shape[1])
            y_f_mesh = np.linspace(0,self.L[1],self.flow[0].shape[0])
            x_f_grid, y_f_grid = np.meshgrid(x_f_mesh,y_f_mesh)
            points = np.array([x_f_grid.flatten(), y_f_grid.flatten()]).T
            self.flow_points = points
        else:
            # time-dependent flow
            x_f_mesh = np.linspace(0,self.L[0],self.flow[0].shape[2])
            y_f_mesh = np.linspace(0,self.L[1],self.flow[0].shape[1])
            x_f_grid, y_f_grid = np.meshgrid(x_f_mesh,y_f_mesh)
            points = np.array([x_f_grid.flatten(), y_f_grid.flatten()]).T
            self.flow_points = points

            # set time
            if not hasattr(tspan, '__iter__'):
                # set flow_times based off zero
                self.flow_times = np.arange(0, tspan*self.flow[0].shape[0], tspan)
            elif len(tspan) == 2:
                self.flow_times = np.linspace(tspan[0], tspan[1], self.flow[0].shape[0])
            else:
                assert len(tspan) == self.flow[0].shape[0]
                self.flow_times = np.array(tspan)



class swarm:

    def __init__(self, swarm_size=100, envir=None, init='random', **kwargs):
        ''' Initalizes planktos swarm in a domain of specified size.

        Arguments:
            envir: environment for the swarm, defaults to the standard environment
            swarm_size: Size of the swarm (int)
            init: Method for initalizing positions.
            kwargs: keyword arguments to be passed to the method for
                initalizing positions

        Methods for initializing the swarm:
            - 'random': Uniform random distribution throughout the domain
            - 'point': All positions set to a single point.
                Required keyword arguments:
                - x = (float) x-coordinate
                - y = (float) y-coordinate
        '''

        # use a new default environment if one was not given
        if envir is None:
            self.envir = environment(init_swarms = self)
        else:
            try:
                assert isinstance(envir,environment)
                envir.swarms.append(self)
                self.envir = envir
            except AssertionError:
                print("Error: invalid environment object.")
                raise

        # initialize bug locations
        self.positions = ma.zeros((swarm_size, 2))
        if init == 'random':
            init_pos.random(self.positions, self.envir.L)
        elif init == 'point':
            init_pos.point(self.positions, kwargs['x'], kwargs['y'])
        else:
            print("Initialization method {} not implemented.".format(init))
            print("Exiting...")
            raise NameError

        # Initialize position history
        self.pos_history = []



    def move(self, dt=1.0, params=None, update_time=True):
        ''' Move all organsims in the swarm over an amount of time dt '''

        # Put current position in the history
        self.pos_history.append(self.positions.copy())

        # Interpolate fluid flow
        if self.envir.flow is None:
            mu = np.array([0, 0])
        else:
            if len(self.envir.flow[0].shape) == 2:
                # temporally constant flow
                mu = self.__interpolate_flow(self.envir.flow, method='cubic')
            else:
                # temporal flow. interpolate in time, and then in space.
                mu = self.__interpolate_flow(self.__interpolate_temporal_flow(),
                                             method='cubic')

        # For now, just have everybody move according to a random walk.
        self.__move_gaussian_walk(self.positions, mu, dt*np.eye(2))

        # Apply boundary conditions.
        self.__apply_boundary_conditions()

        # Record new time
        if update_time:
            self.envir.time_history.append(self.envir.time)
            self.envir.time += dt



    def __apply_boundary_conditions(self):
        '''Apply boundary conditions to self.positions'''

        for dim, bndry in enumerate(self.envir.bndry):

            ### Left boundary ###
            if bndry[0] == 'zero':
                # mask everything exiting on the left
                self.positions[self.positions[:,dim]< 0, :] = ma.masked
            elif bndry[0] == 'noflux':
                # pull everything exiting on the left to 0
                self.positions[self.positions[:,dim]< 0, dim] = 0
            else:
                raise NameError

            ### Right boundary ###
            if bndry[1] == 'zero':
                # mask everything exiting on the right
                self.positions[self.positions[:,dim]> self.envir.L[dim], :] = ma.masked
            elif bndry[1] == 'noflux':
                # pull everything exiting on the left to 0
                self.positions[self.positions[:,dim]> self.envir.L[dim], dim] = self.envir.L[dim]
            else:
                raise NameError



    def __interpolate_flow(self,flow,method):
        '''Interpolate the fluid velocity field at swarm positions'''

        x_vel = interpolate.griddata(self.envir.flow_points, np.ravel(flow[0]), 
                                     self.positions, method=method)
        y_vel = interpolate.griddata(self.envir.flow_points, np.ravel(flow[1]), 
                                     self.positions, method=method)
        return np.array([x_vel, y_vel]).T



    def __interpolate_temporal_flow(self):
        '''Linearly interpolate flow in time'''

        # boundary cases
        if self.envir.time <= self.envir.flow_times[0]:
            return [f[0,:,:] for f in self.envir.flow]
        elif self.envir.time >= self.envir.flow_times[-1]:
            return [f[-1,:,:] for f in self.envir.flow]
        else:
            # linearly interpolate
            indx = np.searchsorted(self.envir.flow_times,self.envir.time)
            diff = self.envir.flow_times[indx] - self.envir.time
            dt = self.envir.flow_times[indx] - self.envir.flow_times[indx-1]
            return [f[indx,:,:]*(dt-diff)/dt + f[indx-1,:,:]*diff/dt
                    for f in self.envir.flow]



    @staticmethod
    def __move_gaussian_walk(pos_array, mean, cov):
        ''' Move all rows of pos_array a random distance specified by
        a gaussian distribution with given mean and covarience matrix.
        
        Arguments:
            pos_array: array to be altered by the gaussian walk
            mean: either a 1-D array mean to be applied to all positions, or
                a 2-D array of means with a number of rows equal to num of positions
            cov: a single covariance matrix'''

        if len(mean.shape) == 1:
            pos_array += np.random.multivariate_normal(mean, 
                            cov, pos_array.shape[0])
        else:
            pos_array += np.random.multivariate_normal(np.zeros(mean.shape[1]), 
                            cov, pos_array.shape[0]) + mean



    def plot(self, blocking=True):
        ''' Plot the current position of the swarm '''

        plt.figure()
        if self.envir.a is not None:
            # add a grassy porous layer background
            grass = np.random.rand(80)*self.envir.L[0]
            for g in grass:
                plt.axvline(x=g, ymax=self.envir.a/self.envir.L[1], color='.5')
        plt.scatter(self.positions[:,0], self.positions[:,1], label='organism')
        plt.xlim((0, self.envir.L[0]))
        plt.ylim((0, self.envir.L[1]))
        plt.title('Organism positions')
        plt.show(blocking)



    def plot_all(self, save_filename=None):
        ''' Plot the entire history of the swarm's movement, incl. current '''

        if len(self.envir.time_history) == 0:
            self.plot()
            return

        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.envir.L[0]), ylim=(0, self.envir.L[1]))
        ax.set_title('Organism positions')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        scat = ax.scatter([], [], label='organism')

        # initialization function: plot the background of each frame
        def init():
            if self.envir.a is not None:
                # add a grassy porous layer background
                grass = np.random.rand(80)*self.envir.L[0]
                for g in grass:
                    ax.axvline(x=g, ymax=self.envir.a/self.envir.L[1], color='.5')
            scat.set_offsets(self.pos_history[0])
            time_text.set_text('time = {:.2f}'.format(self.envir.time_history[0]))
            return scat, time_text

        # animation function. Called sequentially
        def animate(n):
            if n < len(self.pos_history):
                scat.set_offsets(self.pos_history[n])
                time_text.set_text('time = {:.2f}'.format(self.envir.time_history[n]))
            else:
                scat.set_offsets(self.positions)
                time_text.set_text('time = {:.2f}'.format(self.envir.time))
            return scat, time_text

        # infer animation rate from dt between current and last position
        dt = self.envir.time - self.envir.time_history[-1]

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.pos_history)+1,
                                       interval=dt*100, repeat=False, blit=True)
        if save_filename is not None:
            try:
                anim.save(save_filename, dpi=150)
            except:
                print('Failed to save animation.')
                print('Check that you have ffmpeg or mencoder installed!')
                return
        plt.show()
