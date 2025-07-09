'''
Library for different sorts of particle motion, including implementations of
various numerical methods for relevant equations of motion. Most of these will 
take in a Swarm object from which information about the particles
and their environment can be accessed along with relevant parameters. They will 
then return the new particle positions or Delta x after a time dt, depending on
implementation.

Created on Tues Jan 24 2017

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import warnings
import numpy as np
import numpy.ma as ma
from scipy.integrate import solve_ivp



# Decorator to convert an ODE function expecting a 2NxD shaped x-variable array 
#   into a flattened version that can be read into scipy.integrate.ode. All the 
#   ODE generators in this module create 2NxD (or in one special case, NxD) 
#   functions, which is what our built-in solvers expect.
# This decorator should be completely unnecessary unless you really want to use 
#   scipy's integrator.
def flatten_ode(swarm):
    '''Defines a decorator capable of converting a flattened, passed in 
    x-variable array into a 2NxD shape for generated ODE functions, and then 
    take the result of the ODE functions and reflatten. Need knowledge of the 
    dimension of swarm for this, so the Swarm must be passed to the decorator.
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
    

#############################################################################
#                                                                           #
#                           ODE AND SDE SOLVERS!                            #
#                                                                           #
#############################################################################

# TODO: diffusion in porous media https://en.wikipedia.org/wiki/Diffusion#Diffusion_in_porous_media

def RK45(fun, t0, y0, tf, **kwargs):
    '''This is now a wrapper around scipy.integrate.solve_ivp, calling the 
    Runge-Kutta Dormand-Prince [1]_ solver (variable time-step solver) by default.
    It offers a more robust solution than the former, custom implementation.
    
    The passed in ode function (fun) must have call signature (t,x) where x is 
    a 2-D array with a number of columns equal to the spatial dimension.
    The solver will run to tf and then return. It is expected that boundary 
    conditions will be checked after this routine within a Swarm object before 
    the next time step.

    Keyword arguments will be passed to solve_ivp. Important ones include 
    atol and rtol; please see the documentation for scipy.integrate.solve_ivp 
    for further info. Default values: rtol=1e-3 and atol=1e-6.

    Parameters
    ----------
    fun : callable
        Right-hand side of the ODE system. The call signature must be fun(t, y), 
        where t is a scalar (time) and y is an NxD array where D is the 
        dimension of the system.
    t0 : float
        initial time.
    y0 : ndarray of shape (N,D)
        initial state.
    tf : float
        final time, e.g. time to integrate to.

    Returns
    -------
        y : ndarray with shape matching y0
            new state at time tf

    References
    ----------
    .. [1] Dormand, J.R., Prince, P.J. (1980). A family of embedded Runge-Kutta 
       formulae, *Journal of Computational and Applied Mathematics*, 6(1), 19-26.
    '''

    if not isinstance(y0,np.ndarray) or len(y0.shape) != 2:
        raise TypeError("y0 must be an ndarray of shape (N,D)")
    N,D = y0.shape
    vecfun = lambda t,x: fun(t,x.reshape(N,D)).flatten()

    result = solve_ivp(vecfun, (t0, tf), y0.flatten(), vectorized=False, **kwargs)

    return result.y[:,-1].reshape(N,D)



def Euler_brownian_motion(swarm, dt, positions=None, velocities=None, 
                          mu=None, ode=None, sigma=None):
    warnings.simplefilter("ignore", category=SyntaxWarning)
    '''Uses the Euler-Maruyama method to solve the Ito SDE
    
        .. math::
        
            dX_t = \\mu(X_t,t) dt + \\sigma(t) dW_t

    where :math:`\mu` is the drift and :math:`\sigma` is the diffusion.
    :math:`\mu` can be specified directly as a constant or via an ode, or both.
    If both are ommited, the default is to use the mu property of the Swarm
    object plus the local fluid drift. If an ode is given but mu is not, the
    the mu property of the Swarm will be added to the ode velocity term before
    solving (however, the Swarm mu is zero by default).
    
    :math:`\sigma` can be provided a number of ways (see below), and can be 
    dependent on time or agent if passed in directly. This solver is only order 
    0.5 if :math:`\sigma` is dependent on spatial position, so this is not 
    directly supported.

    Parameters
    ----------
    swarm : Swarm object
    dt : float
        time step to take
    positions : NxD ndarray, optional
        the starting positions of all agents that will take the Euler step. If 
        None, the current position of all agents in the Swarm will be used. Note 
        that if positions is provided and mu and/or sigma are not provided, mu 
        and sigma must be the same for all agents in the Swarm - otherwise, if 
        they are properties that differ by agent, it will be impossible to pair 
        positions with corresponding individual values for mu and sigma.
    velocities : NxD ndarray, optional
        the starting velocities of all agents that will take the Euler step. If 
        None, the current velocities of all agents in the Swarm will be used. 
        This is only necessary if an ode is supplied, and it is ignored if 
        positions is None.
    mu : 1D array of length D, array of shape NxD, or array of shape 2NxD, optional
        drift velocity as a 1D array of length D (spatial dimension), an array 
        of shape NxD (where N is the number of agents), or as an array of shape 
        2NxD, where N is the number of agents and D is the spatial dimension. 
        In the last case, the first N rows give the velocity and the second N 
        rows are the acceleration. In this case, a Taylor series method Euler 
        step will be used. If no mu and no ode is given, a default brownian 
        drift of Swarm.get_prop('mu') + the local fluid drift will be used. 
        If mu=None but an ode is given, the default for mu will be 
        Swarm.get_prop('mu') alone, as fluid interaction is assumed to be 
        handled by the ode. Note that the default Swarm value for mu is zeros, 
        so the ode will specify the entire drift term unless mu is set to 
        something else.
    ode : callable, optional
        an ODE function for mu with call signature func(t,x), where t is the 
        current time and x is a 2NxD array of positions/velocities.
        It must return a 2NxD array of velocities/accelerations. See mu for
        information on default behavior if this is not specified.
    sigma : array, optional
        Brownian diffusion coefficient matrix. If None, use the 'cov' property 
        of the Swarm object, or lacking that, the 'D' property. For convienence, 
        :math:`\sigma` can be provided in several ways:

        - As a covariance matrix stored in Swarm.get_prop('cov'). This is the 
          default. The matrix given by this Swarm property is assumed
          to be defined by :math:`\sigma\sigma^T` and independent of time or
          spatial location. The result of using this matrix is that the 
          integrated Wiener process over an interval dt has covariance 
          Swarm.get_prop('cov')*dt, and this will be fed directly into the 
          random number generator to produce motion with these characteristics.
          It should be a square matrix with the length of each side equal to 
          the spatial dimension, and be symmetric.
        - As a diffusion tensor (matrix). The diffusion tensor is given by
          :math:`D = 0.5*\sigma\sigma^T`, so it is really just half the
          covariance matrix. This is the diffusion tensor as given in the
          Fokker-Planck equation or the heat equation. As in the case of the
          covariance matrix, it is assumed constant in time and space, and
          should be specified as a Swarm property with the name 'D'. Again,
          it will be fed directly into the random number generator. 
          It should be a square matrix with the length of each side equal to 
          the spatial dimension, and be symmetric.
        - Given directly as an argument to this function. In this case, it should
          be an DxD array, or an NxDxD array where N is the number of
          agents and D the spatial dimension. Since this is an Euler step 
          solver, sigma is assumed constant across dt.

    Returns
    -------
    NxD array (N number of agents, D spatial dimension)
        New agent positions after the Euler step.
    
    Notes
    -----
    TODO: If sigma is spatially dependent, in order to maintain strong order 1 
    convergence we need the method due to Milstein (1974) (see Kloeden and Platen). 
    This might be useful for turbulant models, or models where the energy of the 
    random walk is dependent upon local fluid properties. This would be 
    particularly useful in the case of media with different densities or in 
    the case of a chemical concentration that excites or depresses behavior. 
    When implementing, the details of this function will need to be changed to
    directly accept a spatially varying sigma. We should implement a 
    Stratonovich function call at that time too, because with a spatially 
    dependent sigma, Ito and Stratonovich are no longer equivalent.
    '''
    warnings.resetwarnings()
    
    # get critical info about number of agents and dimension of domain
    if positions is not None:
        if positions.ndim == 1:
            # one agent case
            positions = np.array([positions,])
        n_agents = positions.shape[0]
        n_dim = positions.shape[1]
    else:
        n_agents = swarm.positions.shape[0]
        n_dim = swarm.positions.shape[1]

    # parse mu and ode arguments
    if mu is None and ode is None:
        # default mu is mean drift plus local fluid velocity
        try:
            stoc_mu = swarm.get_prop('mu') + swarm.get_fluid_drift(positions=positions)
        except ValueError:
            print("Mu must be explicitly passed (or constant among agents).")
            raise

    elif mu is None:
        stoc_mu = swarm.get_prop('mu')
        if stoc_mu.ndim > 1 and positions is not None:
            raise ValueError("Mu must be explicitly passed (or constant among agents).")
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
        if positions is None:
            ode_mu = ode(swarm.envir.time, ma.concatenate((swarm.positions,swarm.velocities)))
        else:
            assert velocities is not None, "Velocities must be specified with ode"
            ode_mu = ode(swarm.envir.time, ma.concatenate((positions,velocities)))
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
            if positions is None:
                positions = swarm.positions
            if not np.isclose(cov.trace(),0):
                # mu already multiplied by dt
                return positions + mu +\
                    swarm.rndState.multivariate_normal(np.zeros(n_dim), cov, n_agents)
            else:
                # mu already multiplied by dt
                return positions + mu
        else: # vector of cov matrices
            if positions is not None:
                assert cov.shape[0] == n_agents, "sigma length must match positions length"
            else:
                positions = swarm.positions
            move = np.zeros_like(positions)
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
            return positions + move
    else:
        # passed in sigma
        if positions is None:
            positions = swarm.positions
        if sigma.ndim == 2: # Single sigma for all agents
            # mu already multiplied by dt
            return positions + mu +\
                sigma @ swarm.rndState.multivariate_normal(np.zeros(n_dim), dt*np.eye(n_dim))
        else: # Different sigma for each agent
            move = np.zeros_like(positions)
            for ii in range(n_agents):
                if mu.ndim > 1:
                    this_mu = mu[ii,:]
                else:
                    this_mu = mu
                move[ii,:] = this_mu +\
                    sigma[ii,...] @ swarm.rndState.multivariate_normal(np.zeros(n_dim), dt*np.eye(n_dim))
            return positions + move



#############################################################################
#                                                                           #
#                        ODE GENERATOR FUNCTIONS                            #
#  These functions generate a handle to an ODE for use within a stochastic  #
#      solver or scipy.integrate.ode (with the flatten_ode decorator).      #
#                                                                           #
#############################################################################

def inertial_particles(swarm):
    warnings.simplefilter("ignore", category=SyntaxWarning)
    '''Function generator for ODEs governing small, rigid, spherical particles 
    whose dynamics can be described by the linearized Maxey-Riley equation [2]_
    described in Haller and Sapsis (2008) [3]_. 
    
        .. math::
            \\frac{d\\mathbf{v}}{dt} &= \\frac{3R}{2}\\frac{D\\mathbf{u}}{Dt} -
            \\mu(\\mathbf{v}-\\mathbf{u}) +
            \\left(1-\\frac{3R}{2}\\right)\\mathbf{g} \\\\
            \\text{with}& \\\\
            \\frac{D\\mathbf{u}}{Dt} &= \\mathbf{u}_t + 
            (\\nabla\\mathbf{u})\\mathbf{u}
    
    Critically, it is assumed that 
    :math:`\mu = R/St` is much greater than 1, where R is the density ratio 
    :math:`R=2\\rho_f/(\\rho_f+2\\rho_p)`, and St is the Stokes number.

    Parameters
    ----------
    swarm : Swarm object

    Returns
    -------
    callable, func(t,x)

    Notes
    -----
    Requires that the following are specified in either Swarm.shared_props
    (if uniform across agents) or Swarm.props (for individual variation):

    - rho or R: particle density or density ratio, respectively.
      if supplying R, 0<R<2/3 corresponds to aerosols, R=2/3 is
      neutrally buoyant, and 2/3<R<2 corresponds to bubbles.
    - diam: diameter of particle

    Requires that the following are specified in the fluid environment:

    - char_L: characteristic length scale for Reynolds number calculation
    - nu: kinematic viscosity
    - U: characteristic fluid speed
    - g: acceleration due to gravity (set by default to 9.80665)
    - rho: fluid density (unless R is specified in swarm)

    References
    ----------
    .. [2] Maxey, M.R. and Riley, J.J. (1983). Equation of motion for a small rigid
      sphere in a nonuniform flow. Phys. Fluids, 26(4), 883-889.
    .. [3] Haller, G. and Sapsis, T. (2008). Where do inertial particles go in
      fluid flows? Physica D: Nonlinear Phenomena, 237(5), 573-583.
    '''
    warnings.resetwarnings()
    
    ##### Check for presence of required physical parameters #####
    try:
        R = swarm.get_prop('R')
    except KeyError:
        try:
            rho_p = swarm.get_prop('rho')
            rho_f = swarm.envir.rho
            R = 2*rho_f/(rho_f+2*rho_p)
        except KeyError:
            raise KeyError("Could not find required physical property R or rho in Swarm object.")

    try:
        a = swarm.get_prop('diam')*0.5 # radius of particles
    except KeyError:
        raise KeyError("Could not find required physical property 'diam' in Swarm object.")

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
        
        Parameters
        ----------
        t : float
            current time
        x : 2NxD array (N number of agents, D spatial dimension)
            particle positions and velocities at time t

        Returns
        ------- 
        a 2NxD array that gives dxdt=v then dvdt
        '''

        N = round(x.shape[0]/2)
        fluid_vel = swarm.get_fluid_drift(t,x[:N])
        DuDt = swarm.get_DuDt(t,x[:N])
        Re = swarm.envir.Re # Reynolds number
        St = 2/9*(a/L)**2*Re # Stokes number
        mu = R/St

        dvdt = 3*R/2*DuDt - mu*(x[N:]-fluid_vel) + (1 - 3*R/2)*g
        return ma.concatenate((x[N:],dvdt))

    # Return equations
    return ODEs

                                                   

def highRe_massive_drift(swarm):
    '''Function generator for ODEs governing high Re massive drift with
    drag as formulated by Nathan et al. [4]_. Assumes Re > 10 with 
    neutrally buoyant particles possessing mass and a known cross-sectional area 
    given by the property cross_sec.

        .. math::
            f_D = \\frac{\\rho C_d A}{m}||\\mathbf{u}-\\mathbf{v}||
            (\\mathbf{u} -\\mathbf{v})

    where :math:`f_D` is the drag force acting on the particle, :math:`C_d` is 
    the drag coefficient, :math:`A` is the cross sectional area, :math:`\\rho` 
    is the fluid density, and :math:`\\mathbf{u}` is the fluid velocity. 
    The system of ODEs are specified so that the acceleration is defined to be 
    :math:`f_D/m`.

    Parameters
    ----------
    swarm : Swarm object

    Returns
    -------
    callable, func(t,x)

    Notes
    -----
    Requires that the following are specified in either Swarm.shared_props
    (if uniform across agents) or Swarm.props (for individual variation):

    - m: mass of each agent
    - Cd: Drag coefficient acting on cross-sectional area
    - cross_sec: cross-sectional area of each agent

    Requires that the following are specified in the fluid environment:

    - rho: fluid density

    References
    ----------
    .. [4] R. Nathan, G.G. Katul, G. Bohrer, A. Kuparinen, M.B. Soons, 
      S.E. Thompson, A. Trakhtenbrot, H.S. Horn (2011). Mechanistic models of 
      seed dispersal by wind. Theoretical Ecology, 4(2), 113-132
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
        
        Parameters
        ----------
        t : float
            current time
        x : 2NxD array (N number of agents, D spatial dimension)
            particle positions and velocities at time t

        Returns
        ------- 
        a 2NxD array that gives dxdt=v then dvdt
        '''

        N = round(x.shape[0]/2)
        fluid_vel = swarm.get_fluid_drift(t,x[:N])

        diff = np.linalg.norm(x[N:]-fluid_vel,axis=1)
        dvdt = (rho*Cd*cross_sec/m**2)*(fluid_vel-x[N:])*np.stack((diff,diff,diff)).T

        return ma.concatenate((x[N:],dvdt))

    # Return equations
    return ODEs



def tracer_particles(swarm, incl_dvdt=True):
    '''Function generator for ODEs describing tracer particles.

    Parameters
    ----------
    swarm : Swarm object
    incl_dvdt : bool, default=True
        Whether or not to include equations for dvdt so that x has shape 2NxD 
        matching most other ODEs (dvdt will just be given as 0).
        If False, will expect an NxD array for x instead of a 2NxD array.

    Returns
    -------
    callable, func(t,x)
    '''

    def ODEs(t,x):
        '''Return ODEs for tracer particles
        
        This x will need to be flattened into a N*D 1D array for use in a scipy 
        solver. A decorator is provided for this purpose.
        
        Parameters
        ----------
        t : float
            current time
        x : 2NxD array (N number of agents, D spatial dimension)
            particle positions and velocities at time t

        Returns
        ------- 
        a 2NxD array that gives dxdt=v then dvdt
        '''

        if not incl_dvdt:
            return swarm.get_fluid_drift(t,x)
        else:
            N = round(x.shape[0]/2)
            return np.concatenate((swarm.get_fluid_drift(t,x[:N]),np.zeros((N,x.shape[1]))))

    # Return equations
    return ODEs
    