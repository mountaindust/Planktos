#! /usr/bin/env python3

'''
Swarm class file, for simulating many individuals at once.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import warnings
import init_pos

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

class environment:

    def __init__(self, Lx=100, Ly=100, x_bndry=None, y_bndry=None,
                 init_swarms=None):
        ''' Initialize environmental variables.

        Arguments:
            Lx: Length of domain in x direction
            Ly: Length of domain in y direction
            x_bndry: [left bndry condition, right bndry condition]
            y_bndry: [low bndry condition, high bndry condition]
            init_swarms: initial swarms in this environment

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

        # swarm list
        if init_swarms is None:
            self.swarms = []
        else:
            if isinstance(init_swarms, list):
                self.swarms = init_swarms
            else:
                self.swarms = [init_swarms]



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
