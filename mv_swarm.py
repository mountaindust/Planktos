#! /usr/bin/env python3

'''
Library of functions for initializing and moving a provided swarm group.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import numpy as np

def init_pos(swarm, func_name, kwargs):
    '''Initialize swarm positions with the correct function'''

    if func_name == init_methods[0]:
        random(swarm, swarm.envir.L)
    elif func_name == init_methods[1]:
        # 'point' requires position data, give as tuple
        # pos=(x,y,z) where z is optional
        assert 'pos' in kwargs, 'point source requires pos key word argument'
        point(swarm, kwargs['pos'])
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

def random(swarm, L):
    '''Uniform random initialization'''

    print('Initializing swarm with uniform random positions...')
    swarm.positions[:,0] = swarm.rndState.uniform(0, L[0], swarm.positions.shape[0])
    swarm.positions[:,1] = swarm.rndState.uniform(0, L[1], swarm.positions.shape[0])
    if len(L) == 3:
        swarm.positions[:,2] = swarm.rndState.uniform(0, L[2], swarm.positions.shape[0])



def point(swarm, pos):
    '''Point source'''

    if len(pos) == 2:
        x,y = pos
        print('Initializing swarm as point source at x={}, y={}'.format(x,y))
        swarm.positions[:,0] = x
        swarm.positions[:,1] = y
    elif len(pos) == 3:
        x,y,z = pos
        print('Initializing swarm as point source at x={}, y={}, z={}'.format(x,y,z))
        swarm.positions[:,0] = x
        swarm.positions[:,1] = y
        swarm.positions[:,2] = z
    else:
        raise RuntimeError('Length of pos argument must be 2 or 3.')



#############################################################################
#                                                                           #
#           PREDEFINED SWARM MOVEMENT BEHAVIOR DEFINED BELOW!               #
#      Each of these must be robust to fixed or varying parameters!         #
#                                                                           #
#############################################################################

def gaussian_walk(swarm, mu, cov, dt):
    ''' Get movement according to a gaussian distribution with given mean 
    and covarience matrix.

    Arguments:
        swarm: swarm.positions (to be read-only in this function!)
        mu: N ndarray or kxN ndarray (k agents, N dim) giving the drift
        cov: NxN ndarray or kxNxN ndarray giving the covariance
        dt: time interval
    '''

    # get critical info about number of agents and dimension of domain
    n_agents = swarm.positions.shape[0]
    n_dim = swarm.positions.shape[1]

    if cov.ndim == 2:
        return swarm.rndState.multivariate_normal(np.zeros(n_dim), dt*cov, 
            n_agents) + dt*mu
    else:
        move = np.zeros_like(swarm.positions)
        for ii in range(n_agents):
            if mu.ndim > 1:
                this_mu = mu[ii,:]
            else:
                this_mu = mu
            move[ii,:] = swarm.rndState.multivariate_normal(
                dt*this_mu, dt*cov[ii,:])
        return move
                                                   


def massive_drift(swarm, dt, net_g=0, high_re=False):
    '''Get drift of the swarm due to background flow assuming massive particles 
    with boyancy accleration net_g'''

    # Get acceleration of each agent in neutral boyancy
    dvdt = swarm.get_projectile_motion(high_re=high_re)
    # Add in accel due to gravity
    dvdt[:,-1] += net_g
    # Solve and return velocity of agents with an Euler step
    return dvdt*dt + swarm.velocity
