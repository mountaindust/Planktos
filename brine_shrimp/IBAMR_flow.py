#! /usr/bin/env python3

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import framework, data_IO

# Intialize environment
envir = framework.environment()

# Import IBMAR data on flow
envir.read_IBAMR3d_vtk_data('data/16towers_Re10_len10.vtk')
print('Domain set to {}.'.format(envir.L))

# envir.add_swarm()
# s = envir.swarms[0]

# print('Moving swarm...')
# for ii in range(240):
#     s.move(0.1)

# #s.plot_all('ex_3d.mp4', fps=20)
# s.plot_all()