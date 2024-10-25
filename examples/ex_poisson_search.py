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
        self.props['searching'] = np.full(self.positions.shape[0], True)

        # Shared rates of changing from searching to moving and moving to searching
        #   per unit time.
        self.shared_props['r_sm'] = 0.25
        self.shared_props['r_ms'] = 0.25

    def get_positions(self, dt, params):

        pass