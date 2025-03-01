#! /usr/bin/env python3
'''
This provides an example of how to handle arbitrary switching times in Planktos. 
It is also an example of a basic intermittent search strategy. 

If you are just starting out, it is suggested that you skip this example and 
come back to it, as it is somewhat complex in its entirety. But there are lots 
of bits and pieces that can be pulled out and used in your own project!
'''

import numpy as np
import matplotlib.pyplot as plt
import planktos

# We will assume that the target is in the middle of the domain with radius 0.5
target_rad = 0.5
target_center = np.array((5,5))

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
        self.shared_props['s'] = 2

        # Set covariance matrix for diffusion
        self.shared_props['cov'] *= 0.01 # identity times 0.01

        #   We need a property for the angle of ballistic motion. Note: 'angle' 
        #   is a special property that interacts with plotting. We don't want 
        #   all agents to have an angle for the purposes of plotting; only the 
        #   ballistic motion ones have a specific angle. So we will use a 
        #   different name for the property.
        self.props['b_angle'] = np.zeros(self.N)

        # Whether or not the agent has found the target or not
        self.props['found'] = np.full(self.N, False)

        # If the agent starts within the target area, it finds it immediately
        #   because they all start out searching
        in_target = self.test_for_target(self.positions)
        self.props.loc[in_target, 'found'] = True

        # Get the first three colors in the default colormap
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        self.searching_color = colors[0]
        self.found_color = colors[1]
        self.moving_color = colors[2]

        # Set agent colors accordingly
        self.props['color'] = np.full(self.N, self.searching_color)
        self.props.loc[in_target, 'color'] = self.found_color

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
        if any(moving):
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
        # Find updated positions while testing to see if the target is found
        #
        found_target = self.props['found'].to_numpy(copy=True)
        new_positions = self.positions.copy()

        ### Those who were moving throughout ###
        full_move = np.logical_and(moving,switch_time == -1)
        if any(full_move):
            new_positions[full_move] += self.shared_props['s']*dt*np.array([
                np.cos(self.props.loc[full_move,'b_angle']),
                np.sin(self.props.loc[full_move,'b_angle'])]).T + \
                dt*self.get_fluid_drift(positions=new_positions[full_move])
        
        ### Those who were moving and then started searching ###
        first_move = np.logical_and(moving,switch_time != -1)
        if any(first_move):
            # First, move to location where state change happened
            start_time = np.column_stack([switch_time[first_move] for ii in range(2)])
            new_positions[first_move] += \
                self.shared_props['s']*dt*start_time*np.array([
                np.cos(self.props.loc[first_move,'b_angle']),
                np.sin(self.props.loc[first_move,'b_angle'])]).T + \
                dt*start_time*self.get_fluid_drift(positions=new_positions[first_move])
            # The agent is now searching. Is the target there?
            found_target[first_move] = self.test_for_target(new_positions[first_move])
            # If the target is found, no more movement. Remove these.
            first_move = np.logical_and(first_move,~found_target)
            # Assuming the target was not found, diffuse for the rest of the 
            #   time period (different for each agent)
            first_move_idx = first_move.nonzero()[0]
            for n in first_move_idx:
                new_positions[n] = planktos.motion.Euler_brownian_motion(
                    self, dt*(1-switch_time[n]), new_positions[n])
            # Is the target there?
            found_target[first_move] = self.test_for_target(new_positions[first_move])
        
            
        ### Those who were searching and then started moving ###
        last_move = np.logical_and(searching,switch_time != -1)
        if any(last_move):
            # Diffuse
            last_move_idx = last_move.nonzero()[0]
            for n in last_move_idx:
                new_positions[n] = planktos.motion.Euler_brownian_motion(
                    self, dt*switch_time[n], new_positions[n])
            # Is the target there?
            found_target[last_move] = self.test_for_target(new_positions[last_move])
            # If the target is found, no more movement. Remove these.
            last_move = np.logical_and(last_move,~found_target)
            # Ballistic motion for the rest of the time step using prev determined angle
            time_left = np.column_stack([(1-switch_time[last_move]) for ii in range(2)])
            new_positions[last_move] += \
                self.shared_props['s']*dt*time_left*np.array([
                np.cos(self.props.loc[last_move,'b_angle']),
                np.sin(self.props.loc[last_move,'b_angle'])]).T + \
                dt*time_left*self.get_fluid_drift(positions=new_positions[last_move])
        
        ### Those who are searching only ###
        full_search = np.logical_and(searching,switch_time == -1)
        if any(full_search):
            # Diffuse
            new_positions[full_search] = planktos.motion.Euler_brownian_motion(
                self, dt, new_positions[full_search])
            # Is the target there?
            found_target[full_search] = self.test_for_target(new_positions[full_search])

        #
        # Update states and search results and return positions
        #

        # Record all changes moving -> searching as searching
        self.props.loc[np.logical_and(moving,switch_time != -1),'searching'] = True
        # Record changes searching -> moving only if target was not found first
        self.props.loc[last_move,'searching'] = False
        # Record new "found target" state and update colors
        self.props['found'] = found_target
        self.props.loc[found_target, 'color'] = self.found_color
        self.props.loc[last_move, 'color'] = self.moving_color
        self.props.loc[first_move, 'color'] = self.searching_color
        # Return new agent positions
        return new_positions

######################################################

# To keep this relatively simple, we will use an environment without background 
#   flow.

envir = planktos.environment(x_bndry='periodic', y_bndry='periodic')
swrm = imsearch(envir=envir, seed=2, store_prop_history=True)

# Create a function that, given an axes object, will plot this target so that we 
#   can visualize it. This does not count as an immersed boundary, even though 
#   we will eventually pass it to the environment object to plot it for us.
def plot_target(ax, args):
    theta = np.linspace(0,2*np.pi,200)
    ax.plot(5+0.5*np.cos(theta),5+0.5*np.sin(theta), 'k', alpha=0.5)

envir.plot_structs.append(plot_target)
envir.plot_structs_args.append(None)

swrm.plot()

dt = 0.1

for ii in range(100):
    swrm.move(dt)

swrm.plot_all()