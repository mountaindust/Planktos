#! /usr/bin/env python3

import sys
sys.path.append('../..') # location of Planktos
sys.path.append('..') # location of plt_cyl
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import numpy as np
import Planktos
import plt_cyl

# Whether or not to show the cylinders based on the mesh data
PLOT_CYL = True

# Intialize environment
envir = Planktos.environment()

# Import IBMAR data on flow
envir.read_IBAMR3d_vtk_data('data/16towers_Re10_len10.vtk')
print('Domain set to {}.'.format(envir.L))

envir.add_swarm()
s = envir.swarms[0]

# Specify amount of jitter (mean, covariance)
# Set std as 1 cm = 0.01 m
s.shared_props['cov'] *= 0.01**2

print('Moving swarm...')
for ii in range(240):
    s.move(0.1)

########## This bit is for plotting the cylinders, if so desired ##########

if PLOT_CYL:
    # get mesh data and translate to new domain
    N_cyl = 16 # number of cylinder files
    for n in range(N_cyl):
        if n<10:
            n_str = '0'+str(n)
        else:
            n_str = str(n)
        plt_cyl.add_cylinders_toplot(envir, 'data/cyl_grids_'+n_str+'.vtk')

##########              Plot!               ###########
s.plot_all('brine_shrimp_IBAMR.mp4', fps=20)
#s.plot_all()
