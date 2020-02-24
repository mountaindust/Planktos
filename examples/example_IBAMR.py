#! /usr/bin/env python3
''' Loads last file in IBAMR test data and moves a swarm in the flow'''

from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import sys
sys.path.append('..')
import Planktos, data_IO, misc


envir = Planktos.environment()
envir.read_IBAMR3d_vtk_dataset('../tests/IBAMR_test_data', start=5, finish=None)
# tile flow in a 3,3 grid
envir.tile_flow(3,3)

# plot cylinder in each tile
misc.add_cylinders_toplot(envir, '../tests/IBAMR_test_data/mesh_db.vtk')

envir.add_swarm()
s = envir.swarms[0]
# amount of jitter (variance)
s.shared_props['cov'] *= 0.0001

print('Moving swarm...')
for ii in range(50):
    s.move(0.1)

s.plot_all()
