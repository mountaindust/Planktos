#! /usr/bin/env python3

'''
Library of functions for initializing a provided swarm structure.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

__author__ = "Christopher Strickland"
__email__ = "wcstrick@live.unc.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import numpy as np

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