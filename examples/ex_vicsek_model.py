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
for download somehow (GBs), it will be available on request. Running this example 
will result in a null model simulation that includes the mesh data but does not 
include the fluid velocity data. 

References
----------
.. [1] Tamas Vicsek, Andras Czirok, Eshel Ben-Jacob, Inon Cohen, Ofer Schochet, 
    (1995). "Novel type of phase transition in a system of self-driven 
    particles," Physical Review Letters, 75(6), 1226-1229.
'''

import sys
sys.path.append('..')
import numpy as np
import planktos

##### Begin by loading the fluid and mesh. #####
# with just this line, we get an empty domain with no flow to test things out
envir = planktos.environment(Lx=1, Ly=0.25, x_bndry='periodic', y_bndry='periodic')


##### Load your fluid data and mesh data here! #####

### To load only the IB2d mesh without the flow: ###
envir.read_IB2d_vertex_data('vicsek_mesh/channel.vertex', res=0.001953125)

### To load both IB2d flow data and mesh. Need to generate flow data first! ###
# envir.read_IB2d_vtk_data('air_cylinder', 5.0e-5, 2000)
# envir.wrap_flow(periodic_dim=[True, True])
# # domain is now 1 x 0.25 m
# envir.read_IB2d_vertex_data('vicsek_mesh/channel.vertex')

# Set the swarm size
SWARM_SIZE = 500

##### Vicsek model of movement #####
# we will assume that in quiescent fluid, all particles move at a constant
#   speed that does not vary between particles. Other parameter will follow the 
#   Vicsek paper as well. One can easily explore the effects of individual 
#   variation by using props instead of shared_props.
class vicsek(planktos.swarm):
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
        # rnd_angles = self.rndState.rand(self.positions.shape[0])*2*np.pi

        ### Bias initial vel toward the right to move toward cylinder ###
        # Add a random perturbation with angle between [-nu/2, nu/2]
        rnd_angles = self.get_prop('nu')*(
            self.rndState.rand(self.positions.shape[0]) - 0.5)
        self.velocities += self.get_prop('v')*np.array([
            np.cos(rnd_angles), np.sin(rnd_angles)]).T


    def get_positions(self, dt, params):
        # Note that the time-step matters: the angle noise and averaging happens
        #   every dt according to the Vicsek model

        # loop through the agents, checking to see which agents are within range 
        #   and averaging their angles
        avg_angles = np.zeros(self.positions.shape[0])
        for n in range(self.positions.shape[0]):
            pos_diff = self.positions - self.positions[n,:]
            dist = np.sqrt(pos_diff[:,0]**2 + pos_diff[:,1]**2)
            avg_angles[n] = np.arctan2(
                np.mean(self.velocities[dist<self.get_prop('r'),1]),
                np.mean(self.velocities[dist<self.get_prop('r'),0]))

        # find new angles according to the Vicsek model
        angle_noise = self.get_prop('nu')*(
            self.rndState.rand(self.positions.shape[0]) - 0.5)
        new_angles = avg_angles + angle_noise

        # convert to velocities and add the fluid velocity
        new_vel = self.get_prop('v')*\
            np.array([np.cos(new_angles), np.sin(new_angles)]).T +\
            self.get_fluid_drift()
        
        # return new positions
        return self.positions + new_vel*dt


# create a swarm with initial conditions behind the cylinder
x_center = 0.1 # +/- 0.025
y_center = 0.125 # +/- 0.075
IC_pos = np.zeros((SWARM_SIZE,2))
IC_pos[:,0] = (np.random.rand(SWARM_SIZE)-0.5)*0.05 + x_center
IC_pos[:,1] = (np.random.rand(SWARM_SIZE)-0.5)*0.15 + y_center

# create Vicsek swarm
swrm = vicsek(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)

# passive particles for comparsion
# swrm = planktos.swarm(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)
# swrm.shared_props['cov'] *= 0.02**2

# conduct simulation
for ii in range(72): # 18 seconds w/ quarter second timesteps
    swrm.move(0.25)

swrm.plot_all(movie_filename='vicsek_null.mp4', fps=4, fluid='vort') # realtime
