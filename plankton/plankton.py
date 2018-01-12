'''
File for defining plankton behavior in flow.

Created on Fri April 07 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
'''

import sys, warnings
sys.path.append('../')
import numpy as np
import agents, mv_swarm

class plankton(agents.swarm):

    def __init__(self, swarm_size=100, envir=None, phys=None, init='random', **kwargs):
        ''' Initalizes plankton in an environment.
        See agents.swarm for further details.

        Arguments:
            envir: environment for plankton, defaults to a standard environment
            swarm_size: Size of the swarm (int)
            init: Method for initalizing positions.
            kwargs: keyword arguments to be passed to the method for
                initalizing positions
        '''
        # Create a suitable environment with no flow for plankton
        if envir is None:
            envir = agents.environment(Lx=100, Ly=100, x_bndry=['zero','zero'],
                                       y_bndry=['noflux','zero'],
                                       Re=1., rho=1000)
        super(plankton, self).__init__(swarm_size, envir, init, **kwargs)

        # Some plankton properties we might want?
        self.energy = 1
        self.memory = None
        


    def update_positions(self, dt, params):
        ''' This method adds plankton behavior.

        Arguments:
            dt: time step to move forward
            params: parameter list to be passed to Gaussian walk
        '''

        # We want to respond to flow somehow. Here is a brief example.
        if self.envir.flow is None:
            # Just do the default thing
            agents.swarm.update_positions(self, dt, params)
        else:
            # 3D?
            DIM3 = (len(self.envir.L) == 3)

            # Parse optional parameters
            if params is not None:
                assert isinstance(params[1], np.ndarray), "cov must be ndarray"
                if not DIM3:
                    assert len(params[0]) == 2, "mu must be length 2"
                    assert params[1].shape == (2,2), "cov must be shape (2,2)"
                else:
                    assert len(params[0]) == 3, "mu must be length 3"
                    assert params[1].shape == (3,3), "cov must be shape (3,3)"
            else:
                params = (0, np.eye(len(self.envir.L)))

            ### Fight against the current to some degree? ###
            fluid_drift = self.get_fluid_drift()
            fluid_mag = np.sqrt(fluid_drift[:,0]**2 + fluid_drift[:,1]**2)
            # plankton resist 3/4 of the current up to 3 m/s total resistance
            resist = -fluid_drift*.75
            resist_mag = np.sqrt(resist[:,0]**2 + resist[:,1]**2)
            resist_over = np.maximum(resist_mag-3,np.zeros_like(resist_mag))
            for dim in range(resist.shape[1]):
                resist[:,dim] *= 3/(resist_over+3)
            # total drift with resistance
            drift = fluid_drift + resist + params[0]

            # Move according to random walk with drift
            mv_swarm.gaussian_walk(self.positions, dt*drift, dt*params[1])
