#! /usr/bin/env python3
'''
This provides an example of how to handle arbitrary switching times in Planktos. 
It is also an example of a basic intermittent search strategy. 

If you are just starting out, it is suggested that you skip this example and 
come back to it, as it is somewhat complex in its entirety. But there are lots 
of bits and pieces that can be pulled out and used in your own project!
'''

import numpy as np
import planktos

# To keep this relatively simple, we will use an environment without background 
#   flow.

envir = planktos.environment()
swrm = planktos.swarm(envir=envir, seed=1, store_prop_history=True)

# We will assume that the target is in the middle of the domain with radius 0.5
target_rad = 0.5
target_center = np.array((5,5))

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

        # Whether or not the agent has found the target or not
        self.props['found'] = np.full(self.N, False)

        # If the agent starts within the target area, it finds it immediately
        in_target = self.test_for_target(self.positions)
        self.props.loc[in_target, 'found'] = True

    @staticmethod
    def test_for_target(positions):
        # Reusable function for testing if an array of positions is within the 
        #   circular target, as defined outside the class.
        return np.linalg.norm(positions-target_center, axis=1) < target_rad

    def get_positions(self, dt, params):

        # Get an array that will code the proprotion of [t,t+dt] at which the 
        #   agents switched state. -1 will mean they didn't switch.
        switch_time = -1*np.ones(self.N)

        # It's best to use the swarm's own rndState object to generate random 
        #   numbers for stochastic processes. That way, everything is 
        #   reproducable with a single seed.

        # Get one random number in [0,1] for each agent
        rand_numbers = self.rndState.random(self.N)

        # We need to copy over the information about which agent is searching 
        #   versus moving. Later, we will update all of them at once.
        searching = self.props['searching'].to_numpy(copy=True)
        moving = ~searching.copy()
        # remove from 'searching' the ones who have already found the target.
        #   They should remain frozen and not update.
        searching[self.props['found']] = False

        #
        # Gather switch times for any ballistic motion agents and transition 
        # them to searching agents.
        #
        if np.any(moving):
            # test random number against CDF of exp dist to see if a switch occurs
            switch_ms = 1-np.exp(-self.shared_props['r_ms']*dt) > rand_numbers[moving]
            # invert CDF to get switch time. switch_ms is a bool, which will 
            #   convert to zero or one when multiplied. The logic his handled 
            #   this way because the length of switch_ms is the same as
            #   switch_time[switch_ms], which is less than switch_time.
            switch_time[moving] += switch_ms*(1-self.shared_props['r_ms']*
                                              np.log(1-rand_numbers[moving]))

        #
        # Gather switch times and angles for searching agents and transition
        # them to moving agents
        #
        if any(searching):
            # test random number against CDF of exp dist to see if a switch occurs
            switch_sm = 1-np.exp(-self.shared_props['r_sm']*dt) > rand_numbers[searching]
            # invert CDF to get switch time
            switch_time[searching] += switch_sm*(1-self.shared_props['r_sm']*
                                                 np.log(1-rand_numbers[searching]))
            # Assign a new random angle to all ballistic motion.
            newangle_agents = searching.copy()
            newangle_agents[searching] = switch_sm # bool of searching AND switching
            # The new angle is uniform random in [0,2pi]
            self.props.loc[newangle_agents,'b_angle'] = \
                2*np.pi*self.rndState.random(np.sum(switch_sm))
            
        #
        # Find updated positions and return them while testing to see if the 
        # target is found
        #
        found_target = self.props['found'].to_numpy(copy=True)
        new_positions = self.positions.copy()

        ### Those who were moving throughout ###
        full_move = np.logical_and(moving,switch_time == -1)
        new_positions[full_move] += self.shared_props['s']*np.array(
            np.cos(self.props.loc[full_move,'b_angle']),
            np.sin(self.props.loc[full_move,'b_angle'])) + \
            self.get_fluid_drift(positions=new_positions[full_move])
        
        ### Those who were moving and then started searching ###
        first_move = np.logical_and(moving,switch_time != -1)
        # First, move to location where state change happened
        new_positions[first_move] += self.shared_props['s']*switch_time*np.array(
            np.cos(self.props.loc[first_move,'b_angle']),
            np.sin(self.props.loc[first_move,'b_angle'])) + \
            self.get_fluid_drift(positions=new_positions[first_move])
        # The agent is now searching. Is the target there?
        in_target = self.test_for_target(new_positions[first_move])
        found_target[first_move] = in_target
        # If the target is found, no more movement
        first_move = np.logical_and(first_move,~found_target)
        # Assuming the target was not found, diffuse for the rest of the time period
        # TODO
        # Is the target there?
        # TODO
            
        ### Those who were searching and then started moving ###
        last_move = np.logical_and(searching,switch_time != -1)

