#! /usr/bin/env python3

'''
Library for different sorts of particle motion, including implementations or
interfaces to various numerical methods for relevant equations of motion. Most
of these will take in a swarm object from which information about the particles
and their environment can be accessed, along with parameters. They will then
return the new particle positions after a time dt, or Delta x depending on
implementation.

Created on Tues Jan 24 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import numpy as np

#############################################################################
#                                                                           #
#           PREDEFINED SWARM MOVEMENT BEHAVIOR DEFINED BELOW!               #
#      Each of these must be robust to fixed or varying parameters!         #
#                                                                           #
#############################################################################

def gaussian_walk(swarm, mu, cov):
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

    if cov.ndim == 2: # Single cov matrix
        if not np.isclose(cov.trace(),0):
            return swarm.rndState.multivariate_normal(np.zeros(n_dim), cov, 
                n_agents) + mu
        else:
            return mu
    else: # vector of cov matrices
        move = np.zeros_like(swarm.positions)
        for ii in range(n_agents):
            if mu.ndim > 1:
                this_mu = mu[ii,:]
            else:
                this_mu = mu
            if not np.isclose(cov[ii,:].trace(),0):
                move[ii,:] = swarm.rndState.multivariate_normal(this_mu, cov[ii,:])
            else:
                move[ii,:] = this_mu
        return move



def inertial_particles(swarm, dt):
    pass
                                                   


def massive_drift(swarm, dt, net_g=0, high_re=False):
    '''Get drift of the swarm due to background flow assuming massive particles 
    with boyancy acceleration net_g.
    Includes drag, inertia, and background flow velocity.

    Arguments:
        dt: time interval
        net_g: net acceleration due to gravity
        high_re: If false (default), assume Re<0.1 for all agents. Otherwise,
        assume Re > 10 for all agents.
    
    TODO: Note that we assume all members of a swarm are approx the same.
    Requires that the following are specified in swarm.shared_props['phys']:
        Cd: Drag coefficient
        S: cross-sectional area of each agent
        m: mass of each agent
        L: diameter of the agent (low Re only)

    Requires that the following are specified in envir:
        rho: fluid density
        mu: dynamic viscosity (if low Re)
    '''

    # Get fluid velocity
    vel = swarm.get_fluid_drift()

    # Check for swarm.shared_props['phys'] and other parameters
    assert 'phys' in swarm.shared_props and\
        isinstance(swarm.shared_props['phys'], dict), "swarm phys not specified"
    for key in ['Cd', 'm']:
        assert key in swarm.shared_props['phys'], "{} not found in swarm phys".format(key)
    assert swarm.envir.rho is not None, "rho not specified"
    if not high_re:
        assert swarm.envir.mu is not None, "mu not specified"
        assert 'diam' in swarm.shared_props, "diameter of agents not specified"
    else:
        assert 'S' in swarm.shared_props['phys'], "Cross-sectional area (S) not found in "+\
        "swarm phys"

    phys = swarm.shared_props['phys']

    # Get acceleration of each agent in neutral boyancy
    if high_re:
        diff = np.linalg.norm(swarm.velocity-vel,axis=1)
        dvdt = swarm.acceleration/phys['m'] -\
        (swarm.envir.rho*phys['Cd']*phys['S']/2/phys['m'])*\
        (swarm.velocity - vel)*np.stack((diff,diff,diff)).T
    else:
        dvdt = swarm.acceleration/phys['m'] -\
        (swarm.envir.mu*phys['Cd']*swarm.shared_props['diam']/2/phys['m'])*\
        (swarm.velocity - vel)

    # Add in accel due to gravity
    dvdt[:,-1] += net_g
    # Solve and return velocity of agents with an Euler step
    return dvdt*dt + swarm.velocity
