#! /usr/bin/env python3

'''
Library of functions for initializing and randomly moving a 
provided swarm structure.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import numpy as np

# list of swarm position initialization functions defined below
init_methods = ['random', 'point']

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