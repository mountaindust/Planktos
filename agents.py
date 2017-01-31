#! /usr/bin/env python3

'''
Swarm class file, for simulating many individuals at once.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import warnings
from math import exp, log
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import init_pos

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class environment:

    def __init__(self, Lx=100, Ly=100, x_bndry=None, y_bndry=None, flow=None,
                 init_swarms=None, Re=None, rho=None):
        ''' Initialize environmental variables.

        Arguments:
            Lx: Length of domain in x direction
            Ly: Length of domain in y direction
            x_bndry: [left bndry condition, right bndry condition]
            y_bndry: [low bndry condition, high bndry condition]
            flow: TBD
            init_swarms: initial swarms in this environment
            Re: Reynolds number of environment (optional)
            rho: fluid density of environment, kh/m**3 (optional)

        Right now, supported boundary conditions are 'zero' (default) and 'noflux'.
        '''

        # Save domain size
        self.L = [Lx, Ly]

        # Parse boundary conditions
        supprted_conds = ['zero','noflux']
        self.bndry = []

        if x_bndry is None:
            # default boundary conditions
            self.bndry.append(['zero', 'zero'])
        elif x_bndry[0] not in supprted_conds or x_bndry[1] not in supprted_conds:
            print("X boundary condition {} not implemented.".format(x_bndry))
            print("Exiting...")
            raise NameError
        else:
            self.bndry.append(x_bndry)
        if y_bndry is None:
            # default boundary conditions
            self.bndry.append(['zero', 'zero'])
        elif y_bndry[0] not in supprted_conds or y_bndry[1] not in supprted_conds:
            print("Y boundary condition {} not implemented.".format(y_bndry))
            print("Exiting...")
            raise NameError
        else:
            self.bndry.append(y_bndry)

        # save flow
        self.flow = flow

        # swarm list
        if init_swarms is None:
            self.swarms = []
        else:
            if isinstance(init_swarms, list):
                self.swarms = init_swarms
            else:
                self.swarms = [init_swarms]

        ##### Fluid Variables #####

        # Re
        self.re = Re
        # Fluid density kg/m**3
        self.rho = rho
        # Characteristic length
        self.char_L = self.L[1]
        # porous region height
        self.a = None



    def set_brinkman_flow(self, alpha, a, res, U, dpdx):
        '''Specify fully developed 2D flow with a porous region.
        Velocity gradient is zero in the x-direction; porous region is the lower
        part of the y-domain (width=a) with an empty region above.

        Arguments:
            alpha: porosity constant
            a: height of porous region
            res: number of points at which to resolve the flow (int)
            U: velocity at top of domain (v in input3d). scalar or list-like.
            dpdx: dp/dx change in momentum constant. scalar or list-like.

        Sets:
            self.flow: [U.size by] res by res ndarray of flow velocity
            self.a = a
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




    def add_swarm(self, swarm_size=100, init='random', **kwargs):
        ''' Adds a swarm into this environment.

        Arguments:
            swarm_size: size of the swarm (int)
            init: Method for initalizing positions.
            kwargs: keyword arguments to be passed to the method for
                initalizing positions
        '''
        self.swarms.append(swarm(swarm_size, self, init, **kwargs))



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

        # Initialize time and history
        self.time = 0.0
        self.time_history = []
        self.pos_history = []



    def move(self, dt=1.0, params=None):
        ''' Move all organsims in the swarm over an amount of time dt '''

        # Put current time/position in the history
        self.time_history.append(self.time)
        self.pos_history.append(self.positions.copy())

        # For now, just have everybody move according to a random walk.
        self.__gaussian_walk(self.positions, [0,0], dt*np.eye(2))

        # Apply boundary conditions.
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
        
        # Record new time
        self.time += dt



    @staticmethod
    def __gaussian_walk(pos_array, mean, cov):
        ''' Move all rows of pos_array a random distance specified by
        a gaussian distribution with given mean and covarience matrix.
        
        Arguments:
            pos_array: array to be altered by the gaussian walk
            mean: either a 1-D mean to be applied to all positions, or
                a 2-D array of means with a number of rows equal to num of positions
            cov: a single covariance matrix'''

        mean = np.array(mean)

        if len(mean.shape) == 1:
            pos_array += np.random.multivariate_normal(mean, 
                            cov, pos_array.shape[0])
        else:
            pos_array += np.random.multivariate_normal(np.zeros(np.shape[1]), 
                            cov, pos_array.shape[0]) + mean



    def plot(self, blocking=True):
        ''' Plot the current position of the swarm '''

        plt.figure()
        plt.scatter(self.positions[:,0], self.positions[:,1], label='organism')
        plt.xlim((0, self.envir.L[0]))
        plt.ylim((0, self.envir.L[1]))
        plt.title('Organism positions')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if blocking:
                plt.show()
            else:
                plt.draw()
                plt.pause(0.001)



    def plot_all(self):
        ''' Plot the entire history of the swarm's movement, incl. current '''

        plt.figure()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for pos, t in zip(self.pos_history, self.time_history):
                plt.scatter(pos[:,0], pos[:,1], label='organism')
                plt.xlim((0, self.envir.L[0]))
                plt.ylim((0, self.envir.L[1]))
                plt.title('Organism positions, time = {:.2f}'.format(t))
                plt.draw()
                plt.pause(0.001)
                plt.clf()
            plt.scatter(self.positions[:,0], self.positions[:,1], label='organism')
            plt.xlim((0, self.envir.L[0]))
            plt.ylim((0, self.envir.L[1]))
            plt.title('Organism positions, time = {:.2f}'.format(self.time))
            plt.draw()
            plt.pause(0.001)
            plt.show()
