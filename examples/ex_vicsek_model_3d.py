#! /usr/bin/env python3
'''
This example is a 3D version of the 2D Vicsek model [1]_ of collective behavior 
(flocking and schooling) implemented in ex_vicsek_model.py. It is a preliminary 
study for future wind tunnel experiments.

The fluid velocity data will be made available on request. Running this example 
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
# setup an empty domain representing one of the two cases

# Wind Tunnel:
# envir = planktos.Environment(Lx=0.21, Ly=2.5, Lz=0.2, x_bndry='noflux', 
#         y_bndry='zero', z_bndry='noflux')

# Periodic Wind Tunnel:
envir = planktos.Environment(Lx=0.21, Ly=2.5, Lz=0.2, x_bndry='noflux', 
        y_bndry='periodic', z_bndry='noflux')

##### Load fluid data #####
try:
    # envir.read_comsol_vtu_data('comsol_data/WindTunnel-p22mps.vtu')
    # envir.center_cell_regrid()
    envir.read_comsol_vtu_data('comsol_data/PeriodicWindTunnel-p22mps.vtu')
    envir.center_cell_regrid(periodic_dim=(False, True, False))
except:
    print('Running null model (without flow)')

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
        self.shared_props['nu_phi'] = 0.25

        # particles will pay attention to other particles within a radius r
        self.shared_props['r'] = 0.1 # constant in Vicsek et al. (r=1)

        ##### Initial conditions #####
        # by default, initial velocities are set to the background flow at the 
        #   initial position or zero if there is no flow.

        ### Uniform random angle IC, just to verify things are working ###
        # rnd_angles_theta = self.rndState.random(self.N)*2*np.pi
        # rnd_angles_phi = self.rndState.random(self.N)*np.pi

        ### Bias initial vel toward y+ to move toward cylinder ###
        # Add a random perturbation with theta angle between [-nu/2, nu/2] and
        #   phi angle between pi/2 + [-nu/4, nu/4]
        rnd_angles_theta = np.pi/2 + self.get_prop('nu_theta')*(
            self.rndState.random(self.N) - 0.5)
        rnd_angles_phi = np.ones(self.N)*np.pi/2
        # rnd_angles_phi = np.pi/2 + 0.5*self.get_prop('nu_phi')*(
        #     self.rndState.random(self.N) - 0.5)

        # Set IC
        self.velocities += self.get_prop('v')*np.array([
            np.cos(rnd_angles_theta)*np.sin(rnd_angles_phi), 
            np.sin(rnd_angles_theta)*np.sin(rnd_angles_phi),
            np.cos(rnd_angles_phi)]).T

    def __calc_dist(self, origin, positions_array):
        ''' A private method that calculates the distance between an origin 
        position and all the postions in postions_array, respecting periodic
        boundary conditions when applicable.
            
            Parameters
            ----------
            origin : np array
                1 by d array where d is the dimension of the environment.
            positions_array : np array
                N by d array where N is the number of agents and d is the 
                    dimension of the environment.
            domain : np array
                1 by d array that has the dimensions of the domain. Give the 
                    dimensions of the environment.
        '''

        diffs = np.zeros(positions_array.shape)

        # get boundary condition type
        x_bndry = self.envir.bndry[0][0]
        y_bndry = self.envir.bndry[1][0]
        z_bndry = self.envir.bndry[2][0]

        # get length of each spatial dimension
        domain = self.envir.L

        x_delta = positions_array[:,0] - origin[0]
        y_delta = positions_array[:,1] - origin[1]
        z_delta = positions_array[:,2] - origin[2]
        
        if x_bndry == 'periodic':
            diffs[:,0] = (x_delta + domain[0]/2) % domain[0] - domain[0]/2
        else:
            diffs[:,0] = x_delta

        if y_bndry == 'periodic':
            diffs[:,1] = (y_delta + domain[1]/2) % domain[1] - domain[1]/2
        else:
            diffs[:,1] = y_delta

        if z_bndry == 'periodic':
            diffs[:,2] = (z_delta + domain[2]/2) % domain[2] - domain[2]/2
        else:
            diffs[:,2] = z_delta

        dist = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2 + diffs[:,2]**2)
            
        return dist

    def get_positions(self, dt, params):
        # Note that the time-step matters: the angle noise and averaging happens
        #   every dt according to the Vicsek model

        # loop through the agents, checking to see which agents are within range 
        #   and averaging their angles
        avg_angles_theta = np.zeros(self.N)
        avg_angles_phi = np.zeros(self.N)
        for n in range(self.N):
            dist = self.__calc_dist(self.positions[n,:], self.positions)
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
            self.rndState.random(self.N) - 0.5)
        angle_noise_phi = 0.5*self.get_prop('nu_phi')*(
            self.rndState.random(self.N) - 0.5)
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
x_center = 0.105 # +/- 0.05
z_center = 0.1 # +/- 0.05
IC_pos = np.zeros((SWARM_SIZE,3))
IC_pos[:,0] = (np.random.rand(SWARM_SIZE)-0.5)*0.1 + x_center
IC_pos[:,1] = 0.9
IC_pos[:,2] = (np.random.rand(SWARM_SIZE)-0.5)*0.1 + z_center

# create Vicsek swarm
swrm = vicsek3d(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)

# passive particles for comparsion
# swrm = planktos.swarm(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)
# swrm.shared_props['cov'] *= 0.02**2 # with jitter
# swrm.shared_props['cov'] *= 0 # without jitter

# conduct simulation
for ii in range(80): # 20 seconds w/ quarter second timesteps (null was 40 sec.)
    swrm.move(0.25)

swrm.plot_all(movie_filename='vicsek3d_Periodic.mp4', fps=4) # realtime

# We can also save our simulation into a vtk file 
# swrm.save_pos_to_vtk('results/vicsek_3D_results', 'vicsek3d_PeriodicPassive')
