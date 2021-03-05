#! /usr/bin/env python3
''' Loads data for a concave, 2D mesh generated in ib2d with time-varying flow.
Simulates agents for the purpose of making sure that the boundary is respected.
'''

from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import numpy as np
import sys
sys.path.append('..')
import Planktos

envir = Planktos.environment()
envir.read_IB2d_vtk_data('data/leaf_data', 1.0e-5, 100, d_start=1)
envir.read_IB2d_vertex_data('data/leaf_data/leaf.vertex', 1.45)

envir.add_swarm(seed=10)
s = envir.swarms[0]
s.positions[89,:] = (0.05, 0.075)
s.positions[95,:] = (0.17, 0.1)
s.shared_props['cov'] *= 0.001

# envir.plot_envir()
# s.plot()

print('Moving swarm...')
for ii in range(550):
    s.move(0.0005)
    
s.plot_all()