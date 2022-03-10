#! /usr/bin/env python3
'''
This example is a 3D version of the 2D Vicsek model [1]_ of collective behavior 
(flocking and schooling) implemented in ex_vicsek_model.py. It is a preliminary 
study for future wind tunnel experiments.

The fluid velocity data will be made available on request. Running this example 
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
# setup an empty domain representing one of the two cases
# Wind Tunnel:
envir = planktos.environment(Lx=0.21, Ly=2.5, Lz=0.2, x_bndry='noflux', 
        y_bndry='zero', z_bndry='noflux')
# Periodic Wind Tunnel:
# envir = planktos.environment(Lx=0.21, Ly=2.5, Lz=0.2, x_bndry='periodic', 
#         y_bndry='zero', z_bndry='noflux')

##### Load fluid data #####
#envir.read_comsol_vtu_data('comsol_data/WindTunnel-p22mps.vtu')

##### Load mesh data #####
# this was in mm...
envir.read_stl_mesh_data('comsol_data/Windtunnel_cylinder.stl', 0.001)

##### Set the swarm size #####
SWARM_SIZE = 500

##### Vicsek 3D model of movement #####
# we will assume that in quiescent fluid, all particles move at a constant
#   speed that does not vary between particles. Other parameters will follow the 
#   Vicsek paper as well. One can easily explore the effects of individual 
#   variation by using props instead of shared_props.
# we will use a spherical coordinate system for our extension. theta is in the
#   x-y plane, phi is the inclination angle down from the positive z-direction. 
#   Phi will be between 0 and pi.
class vicsek3d(planktos.swarm):
    def __init__(self, *args, **kwargs):
        super(vicsek3d, self).__init__(*args, **kwargs)

        ##### Parameter choices #####
        # set particle speed (abs of velocity) in absence of fluid here
        self.shared_props['v'] = 0.02

        # angle noise will be between [-nu_theta/2, nu_theta/2] for theta and 
        #   between [-nu_phi/4, nu_phi/4] for phi
        self.shared_props['nu_theta'] = 0.5 # 0.1 used by Vicsek et al.
        self.shared_props['nu_phi'] = 0

        # particles will pay attention to other particles within a radius r
        self.shared_props['r'] = 0.1 # constant in Vicsek et al. (r=1)

        ##### Initial conditions #####
        # by default, initial velocities are set to the background flow at the 
        #   initial position or zero if there is no flow.

        ### Uniform random angle IC, just to verify things are working ###
        # rnd_angles_theta = self.rndState.rand(self.positions.shape[0])*2*np.pi
        # rnd_angles_phi = self.rndState.rand(self.positions.shape[0])*np.pi

        ### Bias initial vel toward y+ to move toward cylinder ###
        # Add a random perturbation with theta angle between [-nu/2, nu/2] and
        #   phi angle between pi/2 + [-nu/4, nu/4]
        rnd_angles_theta = np.pi/2 + self.get_prop('nu_theta')*(
            self.rndState.rand(self.positions.shape[0]) - 0.5)
        rnd_angles_phi = np.ones(self.positions.shape[0])*np.pi/2
        # rnd_angles_phi = np.pi/2 + 0.5*self.get_prop('nu_phi')*(
        #     self.rndState.rand(self.positions.shape[0]) - 0.5)

        # Set IC
        self.velocities += self.get_prop('v')*np.array([
            np.cos(rnd_angles_theta)*np.sin(rnd_angles_phi), 
            np.sin(rnd_angles_theta)*np.sin(rnd_angles_phi),
            np.cos(rnd_angles_phi)]).T


    def get_positions(self, dt, params):
        # Note that the time-step matters: the angle noise and averaging happens
        #   every dt according to the Vicsek model

        # loop through the agents, checking to see which agents are within range 
        #   and averaging their angles
        avg_angles_theta = np.zeros(self.positions.shape[0])
        avg_angles_phi = np.zeros(self.positions.shape[0])
        for n in range(self.positions.shape[0]):
            pos_diff = self.positions - self.positions[n,:]
            dist = np.sqrt(pos_diff[:,0]**2 + pos_diff[:,1]**2 + pos_diff[:,2]**2)
            avg_angles_theta[n] = np.arctan2(
                np.mean(self.velocities[dist<self.get_prop('r'),1]),
                np.mean(self.velocities[dist<self.get_prop('r'),0]))
            mean_vel_mag = np.mean(np.linalg.norm(
                self.velocities[dist<self.get_prop('r'),:], ord=2, axis=1))
            if mean_vel_mag != 0:
                avg_angles_phi[n] = np.arccos(
                np.mean(self.velocities[dist<self.get_prop('r'),2])/mean_vel_mag)
            else:
                avg_angles_phi[n] = np.pi*0.5

        # find new angles according to the Vicsek model
        angle_noise_theta = self.get_prop('nu_theta')*(
            self.rndState.rand(self.positions.shape[0]) - 0.5)
        angle_noise_phi = np.pi/2 + 0.5*self.get_prop('nu_phi')*(
            self.rndState.rand(self.positions.shape[0]) - 0.5)
        new_angles_theta = avg_angles_theta + angle_noise_theta
        new_angles_phi = avg_angles_phi + angle_noise_phi

        # convert to velocities and add the fluid velocity
        new_vel = self.get_prop('v')*np.array([
            np.cos(new_angles_theta)*np.sin(new_angles_phi), 
            np.sin(new_angles_theta)*np.sin(new_angles_phi),
            np.cos(new_angles_phi)]).T + self.get_fluid_drift()
        
        # return new positions
        return self.positions + new_vel*dt


# create a swarm with initial conditions behind the cylinder
x_center = 0.105 # +/- 0.025
z_center = 0.1 # +/- 0.075
IC_pos = np.zeros((SWARM_SIZE,3))
IC_pos[:,0] = (np.random.rand(SWARM_SIZE)-0.5)*0.05 + x_center
IC_pos[:,1] = 0.1
IC_pos[:,2] = (np.random.rand(SWARM_SIZE)-0.5)*0.15 + z_center

# create Vicsek swarm
swrm = vicsek3d(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)

# passive particles for comparsion
# swrm = planktos.swarm(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)
# swrm.shared_props['cov'] *= 0.02**2

# conduct simulation
for ii in range(72): # 18 seconds w/ quarter second timesteps
    swrm.move(0.25)

swrm.plot_all(movie_filename='vicsek3d_null.mp4', fps=4) # realtime
