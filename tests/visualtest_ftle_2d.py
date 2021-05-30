'''
Script for visual-inspection tests of 2D FTLE
'''

import numpy as np
import sys
sys.path.append('..')
import Planktos

# TODO: get a less-difficult geometry
envir = Planktos.environment()
# envir.read_IB2d_vtk_data('data/channel_cyl', 5.0e-5, 1000, d_start=1)
envir.read_IB2d_vtk_data('data/channel_cyl', 5.0e-5, 1000, d_start=20, d_finish=20)
# envir.read_IB2d_vertex_data('data/channel_cyl/channel.vertex')

# s = envir.add_swarm(900, init='grid', grid_dim=(30,30), testdir='x1')

# Test basic tracer particles on a masked mesh.
# envir.calculate_FTLE((30,30), testdir='x1')
sf, time_list, last_time = envir.calculate_FTLE((512,128),T=0.1,dt=0.001)

envir.plot_2D_FTLE()