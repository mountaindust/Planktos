#! /usr/bin/env python3
'''
This provides an example of how to create an ODE generator for solving equations
of motion in Planktos.
'''

import numpy as np
import planktos

# In creating models of motion for agents, you will often want to specify a
#   system of ODEs which defines the mean-field behavior and/or drift. Once the
#   equations are specified, they can be solved with the RK45 solver in 
#   motion.py for purely deterministic motion or passed to Euler_brownian_motion
#   in motion.py as the drift term in a standard SDE. If you were using scipy
#   to solve a system of ODEs, you would simply define a function which takes
#   in the current time and state variables and returns the derivatives as given
#   by the equations. In Planktos, there is one wrinkle to this: the equations
#   of motion are often parameterized by the fluid velocity field or information
#   about the swarm and its properties. So, the equations need access to the
#   swarm object so that they can get this information. This requires an ODE
#   generator, where an ODE function is created with access to the swarm. It
#   works like this:

def my_ode_generator(swarm):
    '''The ODE generator takes in the swarm object as a parameter and returns
    a function which defines the ODEs'''

    def ODEs(t,x):
        '''Because this function is defined inside my_ode_generator, it has
        access to all the variables in that outer scope... including the swarm
        that was passed in! This means that the call signature can be the 
        required (t,x), and yet the swarm is still accessible.

        Both the RK45 and Euler_brownian_motion solvers expect that x will be
        2NxD in shape, where N is the number of agents (programmatically easy
        to find via swarm.positions.shape[0]) and D is the spatial dimension of
        the system. The reason it is 2N instead of N is so that equations giving
        both velocity and acceleration can be specified. If this is not needed,
        just set the second N equations equal to zero.
        '''

        #Let's create equations of motion defining tracer particles plus a bias
        #   toward the mean position of the swarm, similar to ex_agent_behavior.
        #   We won't use the acceleration equations, so we will concatenate on
        #   zeros.

        # The first half of the rows in x are the current positions of the agents.
        #   The second half are teh current velocities.
        N = round(x.shape[0]/2)
        vel = swarm.get_fluid_drift(t,x[:N])
        # Add a bias toward the mean position
        mean_pos = x[:N].mean(axis=0)
        bias_dir = (mean_pos - x[:N])
        bias_dir /= np.expand_dims(np.linalg.norm(bias_dir, axis=1),1) # normalize
        # Add bias to fluid drift
        vel += bias_dir*0.1

        # concatenate zeros
        return np.concatenate((vel, np.zeros((N,x.shape[1]))))

    # the generator then returns the ODE function
    return ODEs

# Now we follow ex_agent_behavior and override get_positions with this new
#   behavior!

class myswarm(planktos.swarm):

    def get_positions(self, dt, params=None):

        # All we have to do now is use the generator to get the ODEs...
        swrm_odes = my_ode_generator(self)

        # ...and then toss them into the Euler_brownian_motion solver!
        #   For this example, we'll stick with the default jitter given by the 
        #   swarm covariance matrix.
        return planktos.motion.Euler_brownian_motion(self, dt, ode=swrm_odes)

# Now we create the environment/swarm and run it!

envir = planktos.environment(Lx=20, Ly=5, Lz=10, y_bndry=['noflux', 'noflux'],
                             rho=1000, mu=1000)
envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=101)
swrm = myswarm(envir=envir)
swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01

print('Moving swarm...')
for ii in range(240):
    swrm.move(0.1)

num_of_steps = len(envir.time_history) + 1
frames = range(0,num_of_steps,5)

swrm.plot_all(frames=frames)