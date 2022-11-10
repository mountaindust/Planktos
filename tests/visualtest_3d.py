#! /usr/bin/env python3
''' Loads data for a complex (concave and convex) 3D stl mesh from COMSOL.
Simulates agents for the purpose of making sure that the boundary is respected.
'''

import sys
sys.path.append('..')
import planktos

envir = planktos.environment()
envir.read_comsol_vtu_data('data/seafan/sea_fan_data.vtu')
envir.read_stl_mesh_data('data/seafan/sea-fan-piece.stl')

# envir.plot_envir()

s = envir.add_swarm(seed=10)
s.shared_props['cov'] *= 1

### Test for mesh_init ###
# s = envir.add_swarm(init='grid', grid_dim=(30,500,50), testdir='x0')
#######

# s.plot()

# print('Moving swarm...')
# for ii in range(550):
#     s.move(1)

print('Moving swarm...')
for ii in range(100):
    s.move(1)

s.plot_all('sea_fan.mp4', fps=10)
# s.plot_all(frames=range(0,551,5), dist='hist')    