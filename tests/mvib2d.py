'''
Script for testing various things related to moving IB meshes
'''

import numpy as np
import sys
sys.path.append('..')
import planktos

envir = planktos.environment(1,1)
envir.read_IB2d_fluid_data('wobbly_beam/viz_IB2d', 5.0e-5, 10)
envir.read_IB2d_mesh_data('wobbly_beam/viz_IB2d', 5.0e-5, 10)

swrm = planktos.swarm(envir=envir, ib_condition='sticky', seed=1)
swrm.shared_props['cov'] *= 0.01

# swrm.plot()

for ii in range(36):
    swrm.move(0.001)

# Agent number 0 moves through boundary in next time step

for ii in range(64):
    swrm.move(0.001)

swrm.plot_all()