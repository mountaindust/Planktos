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
envir.read_comsol_vtu_data('data/seafan/sea_fan_data.vtu')
envir.read_stl_mesh_data('data/seafan/sea-fan-piece.stl')

# envir.plot_envir()

envir.add_swarm(seed=10)
s = envir.swarms[0]
s.shared_props['cov'] *= 1

# s.plot()

print('Moving swarm...')
for ii in range(550):
    s.move(1)

# s.plot_all(frames=range(0,551,5))    