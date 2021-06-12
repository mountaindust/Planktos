#! /usr/bin/env python3
'''
This is a tutorial and minimal working example of specifying non-default agent 
behavior. This is done by overriding parts of the swarm class.
'''

import sys
sys.path.append('../')
import numpy as np
import planktos

# Agent movement is defined by the get_positions method of the swarm class.
#   By default, the only thing this method does is call
#   planktos.motion.Euler_brownian_motion, which uses an Euler method to solve
#   an Ito SDE describing, by default, basic drift-diffusion. However, you can
#   do whatever you like, including:
#   - Solve some other SDE of the general form described in Euler_brownian_motion
#   - Solve a deterministic system of equations instead, using motion RK45
#   - Write some code to do other things, or some combination of these three.
#   To accomplish this, you need to subclass the swarm class and override the
#   get_positions method. We'll walk through this below.

# Boundary conditions (including collisions with immersed mesh structures) and
#   updating the positions, velocities, and accelerations properties of the 
#   swarm will be handled automatically after the get_positions method returns.
#   So all you need to concentrate on is returning the new agent positions from
#   this function, assuming no boundary or mesh interactions occur.

# It's worth looking through the planktos.motion library to see what's there!
#   This library contains the SDE and RK45 solvers, and has some generators for
#   deterministic equations of motion (which you can copy to create your own).
#   There are also several different methods of the swarm class which can 
#   provide key information for behavior. Examples include:
#   - positions : current positions of all agents
#   - get_prop() : return either shared or individual agent properties
#   - get_fluid_drift() : return the fluid velocity at all agent locations
#   - get_dudt() : return fluid velocity time derivative at all agent locations
#   - get_fluid_gradient() : gradient of the magnitude of fluid velocity at all
#        agent locations

# The subclassing and overriding itself is easy. Here we'll provide an example
#   where agents move toward slower fluid velocities.

# First, we create a new swarm class which inherits everything from the original

class myswarm(planktos.swarm):

    # Now we re-write (called overriding) the get_positions method.
    #   Note that the call signature should remain the same as the original!
    # If you've never written a class method before, the first parameter in
    #   the call signature must always be "self". This refers to the swarm 
    #   object itself and is IMPLICITLY PASSED whenever get_positions is called.
    #   In other words, you would call this method via swrm.get_positions(dt),
    #   and NOT "swrm.get_positions(swrm, dt)". This detail doesn't matter so
    #   much here; the swrm.move method is how we update swarms, and it will do
    #   the business of calling get_positions for us. The main thing to remember
    #   is that if you need any swarm attributes or methods, you should access
    #   them via "self.<method or attribute here>". You'll see this below!
    def get_positions(self, dt, params=None):
        '''New get_positions method that advects agents in the direction of
        slower flow'''

        # First, get the gradient of the fluid velocity magnitude at each agent
        #   location.
        grad = self.get_fluid_gradient()
        # This isn't a unit vector, so let's get something we could divide by to
        #   make it a unit vector.
        denom = np.tile(
                        np.linalg.norm(grad, axis=1),
                        (len(self.envir.L),1)).T

        # If the fluid velocity isn't too fast, let's reduce the drift behavior
        #   toward even slower flows by defining a scaling parameter based on
        #   a threshold.
        thres = 0.1
        scale = np.ones_like(grad)
        scale[denom<1] = (denom[denom<1]-0.1)/0.9
        scale[scale<0] = 0

        # if the norm of the gradient is below a certain amount, call it
        #   undetectable and return zero
        grad_low_idx = denom < thres # mm
        grad[grad_low_idx] = 0
        # catch places where grad == 0 so we don't divide by zero!
        denom[grad_low_idx] = 1

        mvdir = -grad/denom

        # OK! So we have all the information about the movement behavior we 
        #   want. Let's add this to fluid-based advection to get a final drift
        #   vector for each agent, mu.
        mu = self.get_fluid_drift() + mvdir*np.sqrt(25)*scale

        # Finally, we will toss this into the SDE solver to get the resulting
        #   positions, which we return. Jitter will still be based off of the
        #   swarm's covariance matrix, as before.
        return planktos.motion.Euler_brownian_motion(self, dt, mu)



############################

# We have now defined a new swarm class, called myswarm, with our custom 
#   behavior. To use it, we follow the same steps as in previous examples, but
#   create an object out of our new class rather than the swarm class itself.

# Create a 3D environment that is a bit longer in the x-direction and a bit
#   shorter in the y-direction. Also, make the y-boundaries solid to agents.
envir = planktos.environment(Lx=20, Ly=5, Lz=10, y_bndry=['noflux', 'noflux'],
                             rho=1000, mu=1000)
envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=101)

# Now we create a swarm object from our new class. It inherits all methods, 
#   defaults, and options as the original swarm class. But we'll just go with
#   the default here.
swrm = myswarm(envir=envir)
swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01

print('Moving swarm...')
for ii in range(240):
    swrm.move(0.1)

num_of_steps = len(envir.time_history) + 1
frames = range(0,num_of_steps,5)

swrm.plot_all(frames=frames)