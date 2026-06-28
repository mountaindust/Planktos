#! /usr/bin/env python3
''' Loads data for a concave, 2D mesh generated in ib2d with time-varying flow.
Simulates agents for the purpose of making sure that the boundary is respected.
'''

import numpy as np
import sys
sys.path.append('..')
import planktos

envir1 = planktos.Environment()
envir1.read_IB2d_fluid_data('data/leaf_data', 1.0e-5, 100, d_start=1, INUM=None)
### Use to test for boundary crossings ###
envir1.read_IB2d_mesh_data('data/leaf_data/leaf.vertex', 1.45)
envir1.add_vertices_to_static_2D_ibmesh()

class permstick(planktos.Swarm):
    def apply_agent_model(self, dt):
        stick = self.get_prop('stick')
        return np.expand_dims(~stick,1)*super().apply_agent_model(dt) +\
               np.expand_dims(stick,1)*self.positions

### Test for boundary crossings ###
# s = permstick(seed=10, envir=envir)
envir1.add_swarm(seed=10)
s1 = envir1.swarms[0]
s1.positions[89,:] = (0.05, 0.075)
s1.positions[95,:] = (0.17, 0.1)
s1.shared_props['cov'] *= 0.001
# s.props['stick'] = s.ib_collision_idx >= 0
#######

### This is the incorrect mesh for the fluid. Use only for init_grid testing ###
# from give_me_circle_vertices import give_Me_Immersed_Boundary_Geometry
# Nx = len(envir.flow_points[0])
# Ny = len(envir.flow_points[1])
# ds = min(envir.L[0]/(2*Nx),envir.L[1]/(2*Ny))
# give_Me_Immersed_Boundary_Geometry(ds,0.05,np.array(envir.L)/2)
# envir.read_IB2d_mesh_data('circle.vertex', method='proximity')
#######

### Test for mesh_init ###
# s = envir.add_swarm(init='grid', grid_dim=(30,40), testdir='x0')
#######

# envir.plot_envir()
# s.plot()

# print('Moving swarm...')
# for ii in range(300): # 500
#     s.move(0.0005, ib_collisions='sticky')
#     s.props['stick'] = np.logical_or(s.props['stick'],s.ib_collision_idx>=0)

print('Moving orig swarm...')
for ii in range(500):
    s1.move(0.0005)
s1.plot()

# s1.plot_all(movie_filename='leaf_2d_vort_sticky.mp4', figsize=(6,9), fps=30, fluid='vort')

envir2 = planktos.Environment()
envir2.read_IB2d_fluid_data('data/leaf_data', 1.0e-5, 100, d_start=1, INUM=15)
### Use to test for boundary crossings ###
envir2.read_IB2d_mesh_data('data/leaf_data/leaf.vertex', 1.45)
envir2  .add_vertices_to_static_2D_ibmesh()

### Test for boundary crossings ###
# s = permstick(seed=10, envir=envir)
envir2.add_swarm(seed=10)
s2 = envir2.swarms[0]
s2.positions[89,:] = (0.05, 0.075)
s2.positions[95,:] = (0.17, 0.1)
s2.shared_props['cov'] *= 0.001

print('Moving new swarm...')

for ii in range(500):
    s2.move(0.0005)
    # compare maximum difference between fluid velocity fields
    # v1 = envir1.flow(envir2.time)
    # v2 = envir2.flow(envir2.time)
    # rel = np.abs(v1[0][:,:])
    # rel[rel == 0] = 1.0
    # max_diff = np.max(np.abs(v1[0][:,:]-v2[0][:,:])) + \
    #     np.max(np.abs(v1[1][:,:]-v2[1][:,:]))
    # # where is the max difference?
    # max_diff_idx = np.unravel_index(np.argmax(np.abs(v1[0][:,:]-v2[0][:,:])), v1[0].shape)
    # # find the value of v1 at that location
    # v1_max_diff = (np.abs(v1[0][max_diff_idx]), np.abs(v1[1][max_diff_idx]))
    # # if it is very small, set rel diff to zero
    # if v1_max_diff[0] < 1.0e-10 and v1_max_diff[1] < 1.0e-10:
    #     max_rel_diff = 0.0
    # else:
    #     max_rel_diff = max_diff / (v1_max_diff[0] + v1_max_diff[1])
    # print(f'Time: {envir2.time:.5f}, Max fluid vel difference: {max_diff:.5e}')
    # print(f'Time: {envir2.time:.5f}, Rel fluid vel difference: {max_rel_diff:.5e}')
# s2.plot()

# for ii in range(500): # 500
#     s.move(0.0005)

# s.plot()
# for ii in range(5): # 500
#     s.move(0.0005)
#     s.plot()

s2.plot()
# s2.plot_all(movie_filename='leaf_2d_vort_dyload.mp4', figsize=(6,9), fps=30, fluid='vort')
