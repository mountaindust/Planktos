#! /usr/bin/env python3

'''
Library of functions for initializing and moving a provided swarm group.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import numpy as np

def init_pos(swarm, func_name, kwargs):
    '''Initialize swarm positions with the correct function'''

    if func_name == init_methods[0]:
        random(swarm.positions, swarm.envir.L)
    elif func_name == init_methods[1]:
        if 'z' in kwargs:
            zarg = kwargs['z']
        else:
            zarg = None
        point(swarm.positions, kwargs['x'], kwargs['y'], zarg)
    else:
        print("Initialization method {} not implemented.".format(func_name))
        print("Exiting...")
        raise NameError



#############################################################################
#                                                                           #
#   PREDEFINED LIST OF SWARM INITIALIZATION FUNCTIONS SPECIFIED BELOW!      #
#                                                                           #
#############################################################################

init_methods = ['random', 'point']

def random(swarm_pos, L):
    '''Uniform random initialization'''

    print('Initializing swarm with uniform random positions...')
    swarm_pos[:,0] = np.random.uniform(0, L[0], swarm_pos.shape[0])
    swarm_pos[:,1] = np.random.uniform(0, L[1], swarm_pos.shape[0])
    if len(L) == 3:
        swarm_pos[:,2] = np.random.uniform(0, L[2], swarm_pos.shape[0])



def point(swarm_pos, x, y, z):
    '''Point source'''

    if z is None:
        print('Initializing swarm with at x={}, y={}'.format(x,y))
        swarm_pos[:,0] = x
        swarm_pos[:,1] = y
    else:
        print('Initializing swarm with at x={}, y={}, z={}'.format(x,y,z))
        swarm_pos[:,0] = x
        swarm_pos[:,1] = y
        swarm_pos[:,2] = z



#############################################################################
#                                                                           #
#           PREDEFINED SWARM MOVEMENT BEHAVIOR DEFINED BELOW!               #
#                                                                           #
#############################################################################

def gaussian_walk(swarm_pos, mean, cov):
    ''' Move all rows of pos_array a random distance specified by
    a gaussian distribution with given mean and covarience matrix.

    Arguments:
        swarm_pos: array to be altered by the gaussian walk
        mean: either a 1-D array mean to be applied to all positions, or
            a 2-D array of means with a number of rows equal to num of positions
        cov: a single covariance matrix'''

    if len(mean.shape) == 1:
        swarm_pos += np.random.multivariate_normal(mean, cov, swarm_pos.shape[0])
    else:
        swarm_pos += np.random.multivariate_normal(np.zeros(mean.shape[1]),
                                                   cov, swarm_pos.shape[0]) + mean
                                                   


def massive_drift(swarm, dt, net_g=0):
    '''Get drift of the swarm due to background flow assuming massive particles 
    with boyancy accleration net_g'''

    # Get acceleration of each agent in neutral boyancy
    dvdt = swarm.get_projectile_motion()
    # Add in accel due to gravity
    dvdt[:,-1] += net_g
    # Solve and return velocity of agents with an Euler step
    return dvdt*dt + swarm.velocity
