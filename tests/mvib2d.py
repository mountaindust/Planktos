'''
Script for testing various things related to moving IB meshes
'''

import numpy as np
import sys
sys.path.append('..')
import planktos

envir = planktos.Environment(1,1)
envir.read_IB2d_fluid_data('wobbly_beam/viz_IB2d', 5.0e-5, 10)
envir.read_IB2d_mesh_data('wobbly_beam/viz_IB2d', 5.0e-5, 10)

swrm = planktos.swarm(envir=envir, ib_condition='sticky', seed=1)
swrm.shared_props['cov'] *= 0.01

# swrm.plot()

for ii in range(36):
    swrm.move(0.001)

# Agent number 0 hits boundary in next time step

swrm.move(0.001)

# Agent number 0 must be stopped after starting very close to boundary

swrm.move(0.001)

for ii in range(12):
    swrm.move(0.001)

# ibmesh stops after data ends

for ii in range(9):
    swrm.move(0.001)

# agents on right move through on next time step (but agents on left don't??)
# agent index 62

for ii in range(41):
    swrm.move(0.001)

swrm.plot_all('mvib2d.mp4')