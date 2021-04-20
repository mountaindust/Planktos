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
import numpy.ma as ma

# Decorator to convert an ODE function expecting a 2NxD shaped x into a flattened
#   version that can be read into scipy.integrate.ode
def flatten_ode(swarm):
    '''Get a decorator capable of converting a flattened, passed in x into a 
    2NxD shape for the ODE functions, and then take the result of the ODE
    functions and reflatten. Need knowledge of the dimension of swarm for this.
    2N accounts for N equations giving the derivative of position w.r.t time and
    N equations giving the derivative of velocity. D is the dimension of the
    problem (2D or 3D).'''
    dim = swarm.positions.shape[1]
    N_dbl = swarm.positions.shape[0]*2
    def decorator(func):
        def wrapper(t,x):
            result = func(t,np.reshape(x,(N_dbl,dim)))
            return result.flatten()
        return wrapper
    return decorator



def RK45(fun, t0, y0, t_bound, rtol=0.001, atol=1e-06, h=0.1):
    '''Take one Runge-Kutta-Fehlberg (forward) step. fun should be (t,x).'''

    raise NotImplementedError("Still working on this!")

    A = np.array([0,1/4,3/8,12/13,1,1/2])
    B = np.array([[1/4, 0, 0, 0, 0],
                  [3/32, 9/32, 0, 0, 0],
                  [1932/2197, -7200/2197, 7296/2197, 0, 0],
                  [439/216, -8, 3680/513, -845/4104, 0],
                  [-8/27, 2, -3544/2565, 1859/4104, -11/40])
    C = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    T = np.array([-1/360, 0, 128/4275, 2187/75240, -1/50, -2/55])

    step_accepted = False
    eps = atol + rtol*np.max(np.abs(y0))

    while not step_accepted:
        K = []
        K.append(h*fun(t0+A[0]*h, y0))
        K.append(h*fun(t0+A[1]*h, y0+B[0,0]*K[0]))
        K.append(h*fun(t0+A[2]*h, y0+B[1,0]*K[0]+B[1,1]*K[1]))
        K.append(h*fun(t0+A[3]*h, y0+B[2,0]*K[0]+B[2,1]*K[1]+B[2,2]*K[2]))
        K.append(h*fun(t0+A[4]*h, y0+B[3,0]*K[0]+B[3,1]*K[1]+B[3,2]*K[2]+B[3,3]*K[3]))
        K.append(h*fun(t0+A[5]*h, y0+B[4,0]*K[0]+B[4,1]*K[1]+B[4,2]*K[2]+B[4,3]*K[3]+B[4,4]*K[4]))

        K = np.array(K)

        new_y = y0 + C@K
        # insert proper norm here.

    

#############################################################################
#                                                                           #
#           PREDEFINED SWARM MOVEMENT BEHAVIOR DEFINED BELOW!               #
#      Each of these must be robust to fixed or varying parameters!         #
#                                                                           #
#############################################################################

# TODO: diffusion in porous media https://en.wikipedia.org/wiki/Diffusion#Diffusion_in_porous_media


def Euler_brownian_motion(swarm, dt, mu=None, ode=None, sigma=None):
    '''Uses the Euler-Maruyama method to solve the Ito SDE:
    
        :math:`dX_t = \mu dt + \sigma dW_t`

    where :math:`\mu` is the drift and :math:`\sigma` is the diffusion.
    :math:`\mu` can be specified directly as a constant or via an ode, or both.
    If both are left as None, the default is to use the mu property of the swarm
    object plus the local fluid drift. If an ode is given but mu is not, the
    the mu property of the swarm will be added to the ode velocity term before
    solving (however, the swarm mu is zero by default).
    
    :math:`\sigma` can be provided a number of ways (see below), and can be 
    dependent on time and/or position (in addition to agent) if passed in directly.
    However, this solver is only order 0.5 if :math:`\sigma` is dependent on
    spatial position.

    Arguments:
        swarm: swarm object
        dt: time interval
        mu: drift velocity as a 1D array of length D, an array of shape N x D, 
            or as an array of shape 2N x D, where N is the number of agents and 
            D is the spatial dimension. In the last case, the first N rows give 
            the velocity and the second N rows are the acceleration. In this 
            case, a Taylor series method Euler step will be used. If no mu and 
            no ode is given, a default brownian drift of swarm.get_prop('mu') + 
            the local fluid drift will be used. If mu=None but an ode is 
            given, the default for mu will be swarm.get_prop('mu') alone, as 
            fluid interaction is assumed to be handled by the ode. Note that the 
            default swarm value for mu is zeros, in which case the ode will 
            specify the entire drift term.
        ode: (optional) an ODE function for mu with call signature func(t,x),
            where t is the current time and x is a 2N x D array of positions/velocities.
            It must return a 2N x D array of velocities/accelerations.
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

    Note that if sigma is spatially dependent, in order to maintain strong order
    1 convergence we need the method due to Milstein (1974) (see Kloeden and Platen). 
    This might be useful for turbulant models, or models where the energy of the 
    random walk is dependent upon local fluid properties. Since these are more 
    niche scenarios, implementing this here is currently left as a TODO. We
    should implement a Stratonovich function call at that time too, because
    with a spatially dependent sigma, Ito and Stratonovich are no longer
    equivalent.
    '''
    
    # get critical info about number of agents and dimension of domain
    n_agents = swarm.positions.shape[0]
    n_dim = swarm.positions.shape[1]

    # parse mu and ode arguments
    if mu is None and ode is None:
        # default mu is mean drift plus local fluid velocity
        stoc_mu = swarm.get_prop('mu') + swarm.get_fluid_drift()
    elif mu is None:
        stoc_mu = swarm.get_prop('mu')
    else:
        if mu.ndim == 1:
            assert len(mu) == n_dim, "mu must be specified all spatial dimensions"
        else:
            assert mu.shape == (n_agents, n_dim)\
                or mu.shape == (n_agents*2, n_dim),\
                "mu must have shape N_agents x N_dim or 2*N_agents x N_dim"
        stoc_mu = mu

    ##### Take Euler step in deterministic part, possibly with Taylor series #####
    if ode is not None:
        assert callable(ode), "ode must be a callable function with signature ode(t,x)."
        # assume that ode retuns a vector of shape 2NxD
        ode_mu = ode(swarm.envir.time, ma.concatenate((swarm.positions,swarm.velocities)))
        if stoc_mu.ndim == 1 or stoc_mu.shape[0] == n_agents:
            ode_mu[:n_agents] += stoc_mu
        else:
            ode_mu += stoc_mu
        # Take Euler step with Taylor series
        mu = dt*ode_mu[:n_agents] + (dt**2)/2*ode_mu[n_agents:]
    else:
        if stoc_mu.ndim == 1 or stoc_mu.shape[0] == n_agents:
            mu = dt*stoc_mu
        else:
            mu = dt*stoc_mu[:n_agents] + (dt**2)/2*stoc_mu[n_agents:]

    # mu is now a differentiated 1D array of length n_dim or an array of shape 
    #   n_agents x n_dim. We dropped any acceleration terms (no longer needed).

    # Depending on the type of diffusion coefficient/covariance passed in,
    #   do different things to evaluate the diffusion part
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



#############################################################################
#                                                                           #
#                        ODE GENERATOR FUNCTIONS                            #
#  These functions generate a handle to an ODE for use within a stochastic  #
#      solver or scipy.integrate.ode (with the flatten_ode decorator).      #
#                                                                           #
#############################################################################



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
        char_L: characteristic length scale for Reynolds number calculation
        nu: kinematic viscosity
        U: characteristic fluid speed
        g: acceleration due to gravity (set by default to 9.80665)
        rho: fluid density (unless R is specified in swarm)

    References:
        Maxey, M.R. and Riley, J.J. (1983). Equation of motion for a small rigid
            sphere in a nonuniform flow. Phys. Fluids, 26(4), 883-889.
        Haller, G. and Sapsis, T. (2008). Where do inertial particles go in
            fluid flows? Physica D: Nonlinear Phenomena, 237(5), 573-583.
    '''
    
    ##### Check for presence of required physical parameters #####
    try:
        R = swarm.get_prop('R')
    except KeyError:
        rho_p = swarm.get_prop('rho')
        rho_f = swarm.envir.rho
        R = 2*rho_f/(rho_f+2*rho_p)

    a = swarm.get_prop('diam')*0.5 # radius of particles

    assert swarm.envir.char_L is not None, "Characteristic length scale in envir not specified."
    L = swarm.envir.char_L
    assert swarm.envir.U is not None, "Characteristic fluid speed in envir not specified."
    assert swarm.envir.nu is not None, "Kinematic viscosity in envir not specified."
    
    g = swarm.envir.g

    def ODEs(t,x):
        '''Given a current time and array of shape 2NxD, where the first N entries
        are the particle positions and the second N entries are the particle
        velocities and D is the dimension, return a 2NxD array representing the
        derivatives of position and velocity as given by the linearized
        Maxey-Riley equation described in Haller and Sapsis (2008).
        
        This x will need to be flattened into a 2*N*D 1D array for use
            in a scipy solver. A decorator is provided for this purpose.
        
        Returns: 
            a 2NxD array that gives dxdt=v then dvdt

        References:
        Maxey, M.R. and Riley, J.J. (1983). Equation of motion for a small rigid
            sphere in a nonuniform flow. Phys. Fluids, 26(4), 883-889.
        Haller, G. and Sapsis, T. (2008). Where do inertial particles go in
            fluid flows? Physica D: Nonlinear Phenomena, 237(5), 573-583.
        '''

        N = round(x.shape[0]/2)
        fluid_vel = swarm.get_fluid_drift(t,x[:N])
        dudt = swarm.get_dudt(t,x[:N])
        Re = swarm.envir.Re # Reynolds number
        St = 2/9*(a/L)**2*Re # Stokes number
        mu = R/St

        dvdt = 3*R/2*dudt - mu*(x[N:]-fluid_vel) + (1 - 3*R/2)*g
        return ma.concatenate((x[N:],dvdt))

    # Return equations
    return ODEs


                                                   

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

    ##### Get required physical parameters #####
    m = swarm.get_prop('m')
    Cd = swarm.get_prop('Cd')
    assert swarm.envir.rho is not None, "rho (fluid density) not specified"
    rho = swarm.envir.rho
    cross_sec = swarm.get_prop('cross_sec')

    # Get acceleration of each agent in neutral boyancy
    def ODEs(t,x):
        '''Given a current time and array of shape 2NxD, where the first N entries
        are the particle positions and the second N entries are the particle
        velocities and D is the dimension, return a 2NxD array representing the
        derivatives of position and velocity for high Re massive drift.

        This x will need to be flattened into a 2*N*D 1D array for use
            in a scipy solver. A decorator is provided for this purpose.
        
        Returns: 
            a 2NxD array that gives dxdt=v then dvdt
        '''

        N = round(x.shape[0]/2)
        fluid_vel = swarm.get_fluid_drift(t,x[:N])

        diff = np.linalg.norm(x[N:]-fluid_vel,axis=1)
        dvdt = (rho*Cd*cross_sec/2/m)*(fluid_vel-x[N:])*np.stack((diff,diff,diff)).T

        return ma.concatenate((x[N:],dvdt))

    # Return equations
    return ODEs



def tracer_particles(swarm, incl_dvdt=True):
    '''Function generator for ODEs describing tracer particles.

    Arguments:
        swarm: swarm object
        incl_dvdt: whether or not to include equations for dvdt so that x has 
            shape 2NxD matching most other ODEs (dvdt will just be given as 0).
            If False, will produce a pre-flattened version of the ODEs - that is, 
            x will be assumed to be 1D of length N*D.
    '''

    def ODEs(t,x):
        '''Return ODEs for tracer particles
        
        This x will need to be flattened into a N*D 1D array for use
            in a scipy solver. A decorator is provided for this purpose.
        
        Returns: 
            a length N*D or 2NxD array that gives dxdt=v [then dvdt]
        '''

        if not incl_dvdt:
            dim = swarm.positions.shape[1]
            positions = np.reshape(x,(round(len(x)/dim),dim))
            return swarm.get_fluid_drift(t,positions).flatten()
        else:
            N = round(x.shape[0]/2)
            return np.concatenate((swarm.get_fluid_drift(t,x[:N]),np.zeros((N,x.shape[1]))))

    # Return equations
    return ODEs
    