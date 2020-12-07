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

# TODO: diffusion in porous media https://en.wikipedia.org/wiki/Diffusion#Diffusion_in_porous_media


def Euler_brownian_motion(swarm, dt, mu, sigma=None):
    '''Uses the Euler-Maruyama method to solve the Ito SDE:
    
        :math:`dX_t = \mu dt + \sigma dW_t`

    where :math:`\mu` passed in as a parameter. :math:`\sigma` can be provided a 
    number of ways (see below), and can be dependent on time and/or position
    (in addition to agent) if passed in directly.

    Use this function when modeling something other than tracer particles, or
    when modeling particles whose behavior changes based on the fluid velocity.

    Arguments:
        swarm: swarm object
        dt: time interval
        mu: drift velocity as an array of shape N x D, where N is the number of 
            agents and D is the spatial dimension, or as an array of shape
            2N x D, where the first N rows give the velocity and the second
            N rows are the acceleration. In this case, a Taylor series method
            Euler step will be used. Alternatively, an ODE function can be
            passed in for mu with call signature func(t,x), where t is the current
            time and x is a 2N x D array of positions/velocities. It must return
            an N x D or 2N x D array of velocities/accelerations.
        sigma: (optional) brownian diffusion coefficient. If None, use the
            'cov' property of the swarm object, or lacking that, the 'D' property.
            See below.

    Returns:
        New agent positions after Euler step.
    
    For convienence, :math:`\sigma` can be provided in several ways:
        - As a covariance matrix, swarm.get_prop('cov'). This matrix is assumed
            to be given by :math:`\sigma\sigma^T` and independent of time or
            spatial location. The result is that the integrated Wiener process
            over an interval dt has covariance swarm.get_prop('cov')*dt, and this
            will be fed directly into the random number generator to produce
            motion with these characteristics.
            The covariance matrix should have shape n x n, where n is the spatial
            dimension, and be symmetric.
        - As a diffusion tensor (matrix). The diffusion tensor is given by
            :math:`D = 0.5*\sigma\sigma^T`, so it is really just half the
            covariance matrix. This is the diffusion tensor as given in the
            Fokker-Planck equation or the heat equation. As in the case of the
            covariance matrix, it is assumed constant in time and space, and
            should be specified as a swarm property with the name 'D'. Again,
            it will be fed directly into the random number generator. 
            D should be a matrix with shape n x n, where n is the spatial 
            dimension, and it should be symmetric.
        - Given directly as an argument to this function. In this case, it should
            be an n x n array, or an N x n x n array where N is the number of
            agents. Since this is an Euler step solver, sigma is assumed constant
            across dt.

    Note that if sigma is spatially dependent, a superior but similar numerical
    method is that due to Milstein (1974). This might be useful for turbulant
    models, or models where the energy of the random walk is dependent upon
    local fluid properties. Since these are more niche scenarios, implementing
    this here is currently left as a TODO.
    '''
    
    # get critical info about number of agents and dimension of domain
    n_agents = swarm.positions.shape[0]
    n_dim = swarm.positions.shape[1]

    if callable(mu):
        mu = mu(swarm.envir.time, np.vstack((swarm.positions,swarm.velocities)))

    # Take Euler step of deterministic part, possibly with Taylor series method
    if mu.shape[0] == n_agents:
        # velocity data only; regular Euler step
        mu = dt*mu
    else:
        # assume mu.shape[0] == 2N. Use accel data for better accuracy.
        mu = dt*mu[:n_agents] + (dt**2)/2*mu[n_agents:]

    # Depending on the type of diffusion coefficient/covariance passed in,
    #   do different things.

    if sigma is None:
        try:
            # go ahead and multiply cov by dt to get the final covariance for
            #   this time step.
            cov = dt*swarm.get_prop('cov')
        except KeyError:
            try:
                # convert D to cov and multiply by dt
                cov = dt*2*swarm.get_prop('D')
            except KeyError as ke:
                raise type(ke)('When sigma=None, swarm must have either a '+
                               'cov or D property in swarm.shared_props or swarm.props.')
        if cov.ndim == 2: # Single cov matrix
            if not np.isclose(cov.trace(),0):
                # mu already multiplied by dt
                return swarm.positions + mu +\
                    swarm.rndState.multivariate_normal(np.zeros(n_dim), cov, n_agents)
            else:
                # mu already multiplied by dt
                return swarm.positions + mu
        else: # vector of cov matrices
            move = np.zeros_like(swarm.positions)
            for ii in range(n_agents):
                if mu.ndim > 1:
                    this_mu = mu[ii,:]
                else:
                    this_mu = mu
                if not np.isclose(cov[ii,:].trace(),0):
                    move[ii,:] = this_mu +\
                        swarm.rndState.multivariate_normal(np.zeros(n_dim), cov[ii,:])
                else:
                    move[ii,:] = this_mu
            return swarm.positions + move
    else:
        # passed in sigma
        if sigma.ndim == 2: # Single sigma for all agents
            # mu already multiplied by dt
            return swarm.positions + mu +\
                sigma @ swarm.rndState.multivariate_normal(np.zeros(n_dim), np.eye(n_dim))
        else: # Different sigma for each agent
            move = np.zeros_like(swarm.positions)
            for ii in range(n_agents):
                if mu.ndim > 1:
                    this_mu = mu[ii,:]
                else:
                    this_mu = mu
                move[ii,:] = this_mu +\
                    sigma[ii,...] @ swarm.rndState.multivariate_normal(np.zeros(n_dim), np.eye(n_dim))
            return swarm.positions + move



def Euler_brownian_fdrift_motion(swarm, dt, mu=None, sigma=None):
    '''Uses the Euler-Maruyama method to solve the Ito SDE:
    
        :math:`dX_t = (f(X_t,t)+\mu)dt + \sigma dW_t`

    where :math:`f(X_t,t)` is the fluid velocity at the current swarm position
    and :math:`\mu` is the brownian drift given by swarm.get_prop('mu') or passed
    in as a parameter. :math:`\sigma` can be provided a number of ways (see below).
    Both :math:`\mu` and :math:`\sigma` can be dependent on time and/or position
    (in addition to agent) if passed in directly.

    Use this function for really basic behavior that does not depend on the fluid
    and to which you would like fluid drift to be automatically added.

    Arguments:
        swarm: swarm object
        dt: time interval
        mu: (optional) brownian drift as an array of shape n or N x n, where n
            is the spatial dimension and N is the number of agents. Since this
            function takes an Euler step, mu is assumed constant across dt.
            If None, use the mu available as a property of swarm
        sigma: (optional) brownian diffusion coefficient. If None, use the
            'cov' property of the swarm object, or lacking that, the 'D' property.
            See below.

    Returns:
        New agent positions after Euler step.
    
    For convienence, :math:`\sigma` can be provided in several ways:
        - As a covariance matrix, swarm.get_prop('cov'). This matrix is assumed
            to be given by :math:`\sigma\sigma^T` and independent of time or
            spatial location. The result is that the integrated Wiener process
            over an interval dt has covariance swarm.get_prop('cov')*dt, and this
            will be fed directly into the random number generator to produce
            motion with these characteristics.
            The covariance matrix should have shape n x n, where n is the spatial
            dimension, and be symmetric.
        - As a diffusion tensor (matrix). The diffusion tensor is given by
            :math:`D = 0.5*\sigma\sigma^T`, so it is really just half the
            covariance matrix. This is the diffusion tensor as given in the
            Fokker-Planck equation or the heat equation. As in the case of the
            covariance matrix, it is assumed constant in time and space, and
            should be specified as a swarm property with the name 'D'. Again,
            it will be fed directly into the random number generator. 
            D should be a matrix with shape n x n, where n is the spatial 
            dimension, and it should be symmetric.
        - Given directly as an argument to this function. In this case, it should
            be an n x n array, or an N x n x n array where N is the number of
            agents. Since this is an Euler step solver, sigma is assumed constant
            across dt.

    Note that if sigma is spatially dependent, a superior but similar numerical
    method is that due to Milstein (1974). This might be useful for turbulant
    models, or models where the energy of the random walk is dependent upon
    local fluid properties. Since these are more niche scenarios, implementing
    this here is currently left as a TODO.
    '''
    
    # get critical info about number of agents and dimension of domain
    n_agents = swarm.positions.shape[0]
    n_dim = swarm.positions.shape[1]

    # go ahead and multiply mu by dt, since that's all that happens in an
    #   Euler step.
    if mu is None:
        mu = dt*swarm.get_prop('mu')
    else:
        mu = dt*mu

    # Depending on the type of diffusion coefficient/covariance passed in,
    #   do different things.

    if sigma is None:
        try:
            # go ahead and multiply cov by dt to get the final covariance for
            #   this time step.
            cov = dt*swarm.get_prop('cov')
        except KeyError:
            try:
                # convert D to cov and multiply by dt
                cov = dt*2*swarm.get_prop('D')
            except KeyError as ke:
                raise type(ke)('When sigma=None, swarm must have either a '+
                               'cov or D property in swarm.shared_props or swarm.props.')
        if cov.ndim == 2: # Single cov matrix
            if not np.isclose(cov.trace(),0):
                # mu already multiplied by dt
                return swarm.positions + swarm.get_fluid_drift()*dt + mu +\
                    swarm.rndState.multivariate_normal(np.zeros(n_dim), cov, n_agents)
            else:
                # mu already multiplied by dt
                return swarm.positions + swarm.get_fluid_drift()*dt + mu
        else: # vector of cov matrices
            move = np.zeros_like(swarm.positions)
            for ii in range(n_agents):
                if mu.ndim > 1:
                    this_mu = mu[ii,:]
                else:
                    this_mu = mu
                if not np.isclose(cov[ii,:].trace(),0):
                    move[ii,:] = this_mu +\
                        swarm.rndState.multivariate_normal(np.zeros(n_dim), cov[ii,:])
                else:
                    move[ii,:] = this_mu
            return swarm.positions + swarm.get_fluid_drift()*dt + move
    else:
        # passed in sigma
        if sigma.ndim == 2: # Single sigma for all agents
            # mu already multiplied by dt
            return swarm.positions + swarm.get_fluid_drift()*dt + mu +\
                sigma @ swarm.rndState.multivariate_normal(np.zeros(n_dim), np.eye(n_dim))
        else: # Different sigma for each agent
            move = np.zeros_like(swarm.positions)
            for ii in range(n_agents):
                if mu.ndim > 1:
                    this_mu = mu[ii,:]
                else:
                    this_mu = mu
                move[ii,:] = this_mu +\
                    sigma[ii,...] @ swarm.rndState.multivariate_normal(np.zeros(n_dim), np.eye(n_dim))
            return swarm.positions + swarm.get_fluid_drift()*dt + move



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

        This x will need to be flattened into a 2*N*D 1D array for use
            in a scipy solver. A decorator is provided for this purpose.
        
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
