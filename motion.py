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

# Decorator to convert 2NxD ODE function into a flattened version that
#   can be read into scipy.integrate.ode
def flatten_ode(swarm):
    '''Get a decorator capable of converting a flattened, passed in x into a 
    2NxD shape for the ODE functions, and then take the result of the ODE
    functions and reflatten. Need knowledge of the dimension of swarm for this.'''
    dim = swarm.positions.shape[1]
    N_dbl = swarm.positions.shape[0]*2
    def decorator(func):
        def wrapper(t,x):
            result = func(t,np.reshape(x,(N_dbl,dim)))
            return result.flatten()
        return wrapper
    return decorator
    

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



def inertial_particles(swarm):
    '''Function generator for ODEs governing small, rigid, spherical particles 
    whose dynamics can be described by the linearized Maxey-Riley equation 
    described in Haller and Sapsis (2008). Critically, it is assumed that 
    mu = R/St, where R is the density ratio 2*rho_f/(rho_f+2*rho_p) and St is 
    the Stokes number, is much greater than 1.

    Arguments:
        swarm: swarm object

    Requires that the following are specified in either swarm.shared_props
    (if uniform across agents) or swarm.props (for individual variation):
        rho or R: particle density or density ratio, respectively.
            if supplying R, 0<R<2/3 corresponds to aerosols, R=2/3 is
            neutrally buoyant, and 2/3<R<2 corresponds to bubbles.
        diam: diameter of particle

    Requires that the following are specified in the fluid environment:
        TODO

    References:
        Maxey, M.R. and Riley, J.J. (1983). Equation of motion for a small rigid
            sphere in a nonuniform flow. Phys. Fluids, 26(4), 883-889.
        Haller, G. and Sapsis, T. (2008). Where do inertial particles go in
            fluid flows? Physica D: Nonlinear Phenomena, 237(5), 573-583.
    '''
    
    ##### Check for presence of required physical parameters #####
    pass

                                                   

def highRe_massive_drift(swarm):
    '''Function generator for ODEs governing high Re massive drift with
    drag and inertia. Assumes Re > 10 with neutrally buoyant particles
    possessing mass and a known cross-sectional area given by the property
    cross_sec.

    Arguments:
        swarm: swarm object

    Requires that the following are specified in either swarm.shared_props
    (if uniform across agents) or swarm.props (for individual variation):
        m: mass of each agent
        Cd: Drag coefficient
        cross_sec: cross-sectional area of each agent

    Requires that the following are specified in the fluid environment:
        rho: fluid density
    '''

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
    rho = swarm.envir.rho
    if 'cross_sec' in swarm.shared_props:
        cross_sec = swarm.shared_props['cross_sec']
    elif 'cross_sec' in swarm.props['cross_sec']:
        cross_sec = swarm.props['cross_sec'].to_numpy()
    else:
        raise RuntimeError('Property cross_sec not found in swarm.shared_props or swarm.props.')

    # Get acceleration of each agent in neutral boyancy
    def ODEs(t,x):
        '''Given a current time and an array of shape 2NxD, where
        the first N entries are the particle positions and the second N entries
        are the particle velocities with D as the dimension.

        NOTE: This x will need to be flattened into a 2*N*D 1D array for use
            in a scipy solver.
        
        Returns: 
            a 2NxD array that gives dxdt=v then dvdt
        '''

        N = np.round(x.shape[0]/2)
        fluid_vel = swarm.get_fluid_drift(t,x[:N])

        diff = np.linalg.norm(x[N:]-fluid_vel,axis=1)
        dvdt = (rho*Cd*cross_sec/2/m)*(fluid_vel-x[N:])*np.stack((diff,diff,diff)).T

        return np.concatenate(x[N:],dvdt)

    # Return equations
    return ODEs
