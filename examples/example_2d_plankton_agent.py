#! /usr/bin/env python3
'''
This example runs a model with plankton agents whose motion is defined by
overriding parts of the swarm class. This is provided as a brief example of 
non-trivial agent behavior specification and implementation.
'''

import sys
sys.path.append('../')
import numpy as np
import planktos
from planktos import motion

################################################################################
#  Subclass the planktos.swarm class and override get_positions. You can also  #
#   override __init__ if there is setup to be done, i.e. if there are special  #
#   parameters/properties you would like to set. An example of this is below.  #
################################################################################

class plankton(planktos.swarm):

    def __init__(self, swarm_size=100, envir=None, init='random', **kwargs):
        ''' Initalizes plankton in an environment.
        See Planktos.swarm for further details.

        Arguments:
            envir: environment for plankton, defaults to a standard environment
            swarm_size: Size of the swarm (int)
            init: Method for initalizing positions.
            kwargs: keyword arguments to be passed to the method for
                initalizing positions
        '''
        # Create a suitable environment with no flow for plankton
        if envir is None:
            envir = planktos.environment(Lx=10, Ly=10, x_bndry=['zero','zero'],
                                       y_bndry=['noflux','zero'],
                                       mu=1000, rho=1000)

        # You need to have this. It calls the original __init__ method to take
        #   care of the non-custom parts of the setup!
        super(plankton, self).__init__(swarm_size, envir, init, **kwargs)

        # Less jitter
        self.shared_props['cov'] *= 0.01

        # Some plankton properties we might want?
        self.energy = 1
        self.memory = None

    def get_positions(self, dt, params=None):
        ''' This method adds plankton behavior.

        Arguments:
            dt: time step to move forward
            params: parameter list to be passed to Gaussian walk
        '''

        # We want to respond to flow somehow. Here is a brief example.
        if self.envir.flow is None:
            # Just do the default thing
            planktos.swarm.get_positions(self, dt, params)
        else:
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
            drift = fluid_drift + resist + self.get_prop('mu')

            # Return movement according to random walk with drift
            return motion.Euler_brownian_motion(self, dt, drift)



################################################################################
#       Now write the script to create and do something with our swarm!        #
################################################################################

p = plankton(swarm_size=100, init='random')
U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-5,8,3)))
p.envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306,
                          res=100, tspan=[0, 20])

print('Moving plankton...')
for ii in range(50):
    p.move(0.5)

p.plot_all()