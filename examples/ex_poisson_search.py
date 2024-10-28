#! /usr/bin/env python3
'''
This provides an example of how to handle arbitrary switching times in Planktos. 
It is also an example of a basic intermittent search strategy.
'''

import numpy as np
import planktos

# To keep this relatively simple, we will use an environment without background 
#   flow.

envir = planktos.environment()
swrm = planktos.swarm(envir=envir, seed=1, store_prop_history=True)

# We will assume that the target is in the middle of the domain with radius 0.5

# Create a function that, given an axes object, will plot this target so that we 
#   can visualize it. This does not count as an immersed boundary, even though 
#   we will eventually pass it to the environment object to plot it for us.
def plot_target(ax, args):
    theta = np.linspace(0,2*np.pi,200)
    ax.plot(5+0.5*np.cos(theta),5+0.5*np.sin(theta), 'k', alpha=0.5)

envir.plot_structs.append(plot_target)
envir.plot_structs_args.append(None)

# Subclass swarm to create our behavior
class imsearch(planktos.swarm):
    def __init__(self, *args, **kwargs):
        super(imsearch, self).__init__(*args, **kwargs)

        # This property will define whether the agents are searching or moving
        #   e.g., which state they are in at any given time.
        self.props['searching'] = np.full(self.N, True)

        # Shared rates of changing from searching to moving and moving to searching
        #   per unit time.
        self.shared_props['r_sm'] = 0.25
        self.shared_props['r_ms'] = 0.25

        # Speed during ballistic motion
        self.shared_props['s'] = 0.1

        #   We need a property for the angle of ballistic motion. Note: 'angle' 
        #   is a special property that interacts with plotting. We don't want 
        #   all agents to have an angle for the purposes of plotting; only the 
        #   ballistic motion ones have a specific angle. So we will use a 
        #   different name for the property.
        self.props['b_angle'] = np.zeros(self.N)

    def get_positions(self, dt, params):

        switch_time = -1*np.ones(self.N)
        rand_numbers = self.rndState.random(self.N)
        searching = self.props['searching'].to_numpy()
        moving = ~searching

        # The below assumes that if the agents switch states during dt, the time 
        #   at which they do is uniformly distributed.

        # Gather switch times for any ballistic motion agents -> searching
        if np.any(moving):
            switch_ms = 1-np.exp(-self.shared_props['r_ms']*dt) > rand_numbers[moving]
            # invert CDF to get switch time
            switch_time[moving] += switch_ms*(1-self.shared_props['r_ms']*
                                              np.log(1-rand_numbers[moving]))

        # Gather switch times for searching agents -> ballistic motion
        if any(searching):
            switch_sm = 1-np.exp(-self.shared_props['r_sm']*dt) > rand_numbers[searching]
            switch_time[searching] += switch_sm*(1-self.shared_props['r_sm']*
                                                 np.log(1-rand_numbers[searching]))
            # Assign a random angle to all ballistic motion.
            newangle_agents = searching.copy()
            newangle_agents[searching] = switch_sm
            self.props.loc[newangle_agents,'b_angle'] = \
                2*np.pi*self.rndState.random(np.sum(switch_sm))
