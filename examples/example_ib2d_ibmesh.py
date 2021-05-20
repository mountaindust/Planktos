#! /usr/bin/env python3
''' Loads data from IB2d Example_Channel_Flow/Example_Flow_Around_Cylinder example 
for flow field AND loads the .vertex data, creating an ibmesh out of it by
attaching nearby points. Agents should not be able to move through the cylinder 
or the channel boundaries.'''

import numpy as np
import sys
sys.path.append('..')
import Planktos


envir = Planktos.environment()
# read in vtk data with dt=5.0e-5, print_dump=1000. Start on dump 1.
envir.read_IB2d_vtk_data('ib2d_data', 5.0e-5, 1000, d_start=1)
# read in vertex data, attaching nearby points
envir.read_IB2d_vertex_data('ib2d_data/channel.vertex')

# tile flow in a 3,3 grid
# envir.tile_flow(3,3)

envir.add_swarm(init=(envir.L[0]*0.1,envir.L[1]*0.5))
s = envir.swarms[0]

# amount of jitter (variance)
s.shared_props['cov'] *= 0.0001

print('Moving swarm...')
for ii in range(46):
    s.move(0.025)

s.plot_all(movie_filename='channel_flow_ibmesh.mp4',fps=3)
