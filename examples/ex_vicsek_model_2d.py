#! /usr/bin/env python3
'''
This example implements the classic Vicsek model [1]_ of collective behavior 
(flocking and schooling) as a biologically interesting example of agent 
interaction with, not only each other (as in the original Vicsek et al. paper), 
but also with the fluid environment they are immersed in and an immersed 
structure which influences both the fluid and the organisms. Most studies of 
collective behavior in self-driven particles assume a quiescent fluid 
environment without structures the particles have to navigate. Planktos is well 
positioned to aid in research targeted at how flocks and schools persist and 
react to environmental conditions.

The 2D fluid velocity data for this example is large and several instances of 
fluid velocity fields were tested. Rather than trying to include all this data 
for download somehow (400 MB), it will be available on request. Running this example 
will result in a null model simulation that includes the mesh data but does not 
include the fluid velocity data.

Special thanks to Kimberlyn Eversman for adding a distance function that 
respects periodic boundary conditions!

References
----------
.. [1] Tamas Vicsek, Andras Czirok, Eshel Ben-Jacob, Inon Cohen, Ofer Schochet, 
    (1995). "Novel type of phase transition in a system of self-driven 
    particles," Physical Review Letters, 75(6), 1226-1229.
'''

import numpy as np
import planktos

##### Begin by loading the fluid and mesh. #####
# with just this line, we get an empty domain with no flow to test things out
envir = planktos.Environment(Lx=1, Ly=0.25, x_bndry='periodic', y_bndry='periodic')


##### Load your fluid data and mesh data here! #####

### To load only the IB2d mesh without the flow: ###
envir.read_IB2d_mesh_data('vicsek_mesh/channel.vertex', method='proximity', 
                          res=0.001953125)

### To load both IB2d flow data and mesh. Need to generate flow data first! ###
# envir.read_IB2d_vtk_data('air_cylinder', 5.0e-5, 2000)
# # domain is now 1 x 0.25 m
# envir.read_IB2d_vertex_data('vicsek_mesh/channel.vertex')

# Set the swarm size
SWARM_SIZE = 500

##### Vicsek model of movement #####
# we will assume that in quiescent fluid, all particles move at a constant
#   speed that does not vary between particles. Other parameter will follow the 
#   Vicsek paper as well. One can easily explore the effects of individual 
#   variation by using props instead of shared_props.
class vicsek(planktos.Swarm):
    def __init__(self, *args, **kwargs):
        super(vicsek, self).__init__(*args, **kwargs)

        ##### Parameter choices #####
        # set particle speed (abs of velocity) in absence of fluid here
        self.shared_props['v'] = 0.02

        # angle noise will be between [-nu/2, nu/2]
        self.shared_props['nu'] = 0.5 # 0.1 used by Vicsek et al.

        # particles will pay attention to other particles within a radius r
        self.shared_props['r'] = 0.1 # constant in Vicsek et al. (r=1)

        ##### Initial conditions #####
        # by default, initial velocities are set to the background flow at the 
        #   initial position or zero if there is no flow.

        ### Uniform random angle IC, just to verify things are working ###
        # rnd_angles = self.rndState.random(self.N)*2*np.pi

        ### Bias initial vel toward the right to move toward cylinder ###
        # Add a random perturbation with angle between [-nu/2, nu/2]
        rnd_angles = self.get_prop('nu')*(
            self.rndState.random(self.N) - 0.5)
        self.velocities += self.get_prop('v')*np.array([
            np.cos(rnd_angles), np.sin(rnd_angles)]).T
    
    def __calc_dist(self, origin, positions_array):
        ''' A private method that calculates the distance between an origin 
        position and all the postions in postions_array, respecting periodic
        boundary conditions when applicable.
            
            Parameters
            ----------
            origin : ndarray
                length d array where d is the dimension of the environment.
            positions_array : np array
                N by d array where N is the number of agents and d is the 
                    dimension of the environment.
            domain : ndarray
                length d array that specifies the dimensions of the environment.
        '''

        diffs = np.zeros(positions_array.shape)

        # get boundary condition type
        x_bndry = self.envir.bndry[0][0]
        y_bndry = self.envir.bndry[1][0]

        # get length of each spatial dimension
        domain = self.envir.L

        x_delta = positions_array[:,0] - origin[0]
        y_delta = positions_array[:,1] - origin[1]
        
        if x_bndry == 'periodic':
            diffs[:,0] = (x_delta + domain[0]/2) % domain[0] - domain[0]/2
        else:
            diffs[:,0] = x_delta

        if y_bndry == 'periodic':
            diffs[:,1] = (y_delta + domain[1]/2) % domain[1] - domain[1]/2
        else:
            diffs[:,1] = y_delta

        dist = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)

        return dist


    def get_positions(self, dt, params):
        # Note that the time-step matters: the angle noise and averaging happens
        #   every dt according to the Vicsek model

        # loop through the agents, checking to see which agents are within range 
        #   and averaging their angles
        avg_angles = np.zeros(self.N)
        for n in range(self.N):
            dist = self.__calc_dist(self.positions[n,:], self.positions)
            avg_angles[n] = np.arctan2(
                np.mean(self.velocities[dist<self.get_prop('r'),1]),
                np.mean(self.velocities[dist<self.get_prop('r'),0]))

        # find new angles according to the Vicsek model
        angle_noise = self.get_prop('nu')*(
            self.rndState.random(self.N) - 0.5)
        new_angles = avg_angles + angle_noise

        # convert to velocities and add the fluid velocity
        new_vel = self.get_prop('v')*\
            np.array([np.cos(new_angles), np.sin(new_angles)]).T +\
            self.get_fluid_drift()
        
        # return new positions
        return self.positions + new_vel*dt


# create a Swarm with initial conditions behind the cylinder
x_center = 0.1 # +/- 0.025
y_center = 0.125 # +/- 0.075
IC_pos = np.zeros((SWARM_SIZE,2))
IC_pos[:,0] = (np.random.rand(SWARM_SIZE)-0.5)*0.05 + x_center
IC_pos[:,1] = (np.random.rand(SWARM_SIZE)-0.5)*0.15 + y_center

# create Vicsek Swarm
swrm = vicsek(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)

# passive particles for comparsion
# swrm = planktos.Swarm(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)
# swrm.shared_props['cov'] *= 0.02**2

# conduct simulation
for ii in range(72): # 18 seconds w/ quarter second timesteps
    swrm.move(0.25)

swrm.plot_all(movie_filename='vicsek_null.mp4', fps=4, fluid='vort') # realtime
