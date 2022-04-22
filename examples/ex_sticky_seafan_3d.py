#! /usr/bin/env python3
'''
This example builds on the ex_ib2d_sticky example by going to 3D and building an
array of cylinders out of a single cylinder (through tiling). The CDF simulation 
was carried out in COMSOL using a thin domain in the x-direction with periodic 
boundary condition. The fluid and cylinder mesh is tiled in this direction within 
Planktos and the cylinders are made to be sticky in order to learn something 
about sea fans.

NOTE: IN ORDER TO RUN THIS EXAMPLE, YOU MUST HAVE THE REQUIRED DATA!
It can be downloaded from: 
https://drive.google.com/drive/folders/1etoBspZ76mwFZ63V0NdqdeVNNKlPYUxu?usp=sharing
Put it into a comsol_data folder in this example directory, and you should be 
good to go!
'''

import sys
sys.path.append('..')
import numpy as np
import planktos

# Begin by loading the fluid and mesh.
envir = planktos.environment()
# In this data, space is in mm but velocity is in m/s. Convert velocity to mm/s.
envir.read_comsol_vtu_data('comsol_data/Velocity2to1_8mmps.vtu', vel_conv=1000)
envir.units = 'mm'
# NOTE: This fluid data is 2.9 x 99.9 x 39.9 mm because it is calculated at the 
#   center of each mesh cell. To fix this, we use the center_cell_regrid 
#   function. This must be done before loading any stl files.
envir.center_cell_regrid(periodic_dim=(True, True, True))

envir.read_stl_mesh_data('comsol_data/seafan_cylinder.stl')
# Tile the fluid and cylinder in the x-direction so that the length of the x- 
#   and z-dimensions are roughly the same.
envir.tile_flow(x=13)

# See ex_ib2d_sticky for details about this "sticky" behavior!
class permstick(planktos.swarm):
    def get_positions(self, dt, params):
        stick = self.get_prop('stick')
        all_move = planktos.motion.Euler_brownian_motion(self, dt)
        return np.expand_dims(~stick,1)*all_move +\
               np.expand_dims(stick,1)*self.positions

# Set the swarm size
SWARM_SIZE = 1000

# Now we create the swarm similar to ex_ib2d_sticky.py. Can possibly do this 
#   in a loop to get aggregate results.

for trial in range(100):
    envir.reset(rm_swarms=True)

    ##### Point IC #####
    # swrm = permstick(swarm_size=SWARM_SIZE, envir=envir, 
    #                  init=(envir.L[0]*0.5,envir.L[1]*0.1,envir.L[2]*0.5))
    ####################

    ###### Uniformly distributed IC in the slice [0.1L_x,0.9L_x] x 0.1L_y x [0.1L_z,0.9L_z] #####
    xz_rnd = np.random.rand(SWARM_SIZE,2)
    IC_pos = np.zeros((SWARM_SIZE,3))
    IC_pos[:,0] = xz_rnd[:,0]*envir.L[0]*0.8 + envir.L[0]*0.1
    IC_pos[:,1] = envir.L[1]*0.1
    IC_pos[:,2] = xz_rnd[:,1]*envir.L[2]*0.8 + envir.L[2]*0.1
    swrm = permstick(swarm_size=SWARM_SIZE, envir=envir, init=IC_pos)
    ####################

    # Set jitter to have a std of 0.5 mm/sec
    swrm.shared_props['cov'] *= 0.25
    swrm.props['stick'] = np.full(SWARM_SIZE, False) # creates a length 1000 array of False

    # Move the swarm similar to as in ex_ib2d_sticky.
    for ii in range(120): # 1 minute w/ half second timesteps
        swrm.move(0.5, ib_collisions='sticky')
        swrm.props['stick'] = np.logical_or(swrm.props['stick'], swrm.ib_collision)

    print('{} organisms, {} stuck ({}%).'.format(SWARM_SIZE, swrm.props['stick'].sum(), 
                                                swrm.props['stick'].sum()/SWARM_SIZE*100))

    # print result to file
    with open('results/seafan2to1_8mmps_sticky_1000dist_stuckfrac.txt', 'a') as f:
        print(swrm.props['stick'].sum()/SWARM_SIZE, file=f)


swrm.plot_all(movie_filename='results/seafan2to1_8mmps_sticky_1000dist.mp4', fps=4) # double-time movie


