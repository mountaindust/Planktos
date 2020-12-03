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
                                                   


def highRe_massive_drift(swarm, net_g=0):
    '''Get drift of the swarm due to background flow assuming Re > 10 with 
    neutrally bouyant massive particles and net acceleration due to gravity net_g.
    Includes drag, inertia, and background flow velocity.

    Arguments:
        dt: time interval
        net_g: net acceleration due to gravity

    Requires that the following are specified in either swarm.shared_props
    (if uniform across agents) or swarm.props (for individual variation):
        m: mass of each agent
        Cd: Drag coefficient
        cross_sec: cross-sectional area of each agent

    Requires that the following are specified in the fluid environment:
        rho: fluid density
    '''

    # Get fluid velocity
    vel = swarm.get_fluid_drift()

    ##### Check for presence of required physical parameters #####
    if 'm' in swarm.shared_props:
        m = swarm.shared_props['m']
    elif 'm' in swarm.props['m']:
        m = swarm.props['m'].to_numpy()
    else:
        raise RuntimeError('Property m not found in swarm.shared_props or swarm.props.')
    if 'Cd' in swarm.shared_props:
        Cd = swarm.shared_props['Cd']
    elif 'Cd' in swarm.props['Cd']:
        Cd = swarm.props['Cd'].to_numpy()
    else:
        raise RuntimeError('Property Cd not found in swarm.shared_props or swarm.props.')
    assert swarm.envir.rho is not None, "rho (fluid density) not specified"
    if 'cross_sec' in swarm.shared_props:
        cross_sec = swarm.shared_props['cross_sec']
    elif 'cross_sec' in swarm.props['cross_sec']:
        cross_sec = swarm.props['cross_sec'].to_numpy()
    else:
        raise RuntimeError('Property cross_sec not found in swarm.shared_props or swarm.props.')

    # Get acceleration of each agent in neutral boyancy
    diff = np.linalg.norm(swarm.velocity-vel,axis=1)
    dvdt = (swarm.envir.rho*Cd*cross_sec/2/m)*\
    (vel - swarm.velocity)*np.stack((diff,diff,diff)).T

    # Add in accel due to gravity
    dvdt[:,-1] += net_g

    # Solve and return velocity of agents with an Euler step
    return dvdt*dt + swarm.velocity
