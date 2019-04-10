#! /usr/bin/env python3
'''
Read IBFE data for 3D cylinder model. Reproduce flowtank experiment with
random walk agents.
'''

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import Planktos, data_IO

# Whether or not to show the cylinders based on the mesh data
PLOT_MODEL = False

# Intialize environment
envir = Planktos.environment()

# Import IBMAR data on flow
print('Reading VTK data. This will take a while...')
envir.read_IBAMR3d_vtk_data('data/RAW_10x20x2cm_8_5s.vtk')
print('Domain set to {} mm.'.format(envir.L))
print('Flow mesh is {}.'.format(envir.flow[0].shape))
print('-------------------------------------------')
# Domain should be 80x320x80 mm
# Model sits (from,to): (2.5,77.5), (85,235), (0.5,20.5)
# Need: 182 mm downstream from model to capture both zones

# Extend domain downstream so Y is 420mm total length
envir.extend(y_plus=100)
print('Domain extended to {} mm'.format(envir.L))
print('Flow mesh is {}.'.format(envir.flow[0].shape))
# NOW:
# Domain should be 80x420x80 mm
# Model sits (from,to): (2.5,77.5), (85,235), (0.5,20.5)
model_bounds = (2.5,77.5,85,235,0,20.5)
print('-------------------------------------------')

# Add swarm right in front of model
s = envir.add_swarm(swarm_s=1000, init='point', pos=(40,84,1))

# Specify amount of jitter (mean, covariance)
# Set sigma**2 as 0.5cm**2/sec =  
# (sigma**2=2*D, D for brine shrimp given in Kohler, Swank, Haefner, Powell 2010)
shrimp_walk = ([0,0,0], 50*np.eye(3))

print('Moving swarm...')
for ii in range(240):
    s.move(0.1, shrimp_walk)

########## This bit plots the model as a translucent rectangle ##########

def plot_model_rect(ax3d, bounds):
    '''Plot the model as a translucent rectangular prism

    Arguments:
        ax3d: Axes3D object
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)'''
    x_range = bounds[0:2]
    y_range = bounds[2:4]
    z_range = bounds[4:]

    xx, yy = np.meshgrid(x_range, y_range)
    ax3d.plot_wireframe(xx, yy, z_range[0], color="g")
    ax3d.plot_surface(xx, yy, z_range[0], color="g", alpha=0.2)
    ax3d.plot_wireframe(xx, yy, z_range[1], color="g")
    ax3d.plot_surface(xx, yy, z_range[1], color="g", alpha=0.2)

    yy, zz = np.meshgrid(y_range, z_range)
    ax3d.plot_wireframe(x_range[0], yy, zz, color="g")
    ax3d.plot_surface(x_range[0], yy, zz, color="g", alpha=0.2)
    ax3d.plot_wireframe(x_range[1], yy, zz, color="g")
    ax3d.plot_surface(x_range[1], yy, zz, color="g", alpha=0.2)

    xx, zz = np.meshgrid(x_range, z_range)
    ax3d.plot_wireframe(xx, y_range[0], zz, color="g")
    ax3d.plot_surface(xx, y_range[0], zz, color="g", alpha=0.2)
    ax3d.plot_wireframe(xx, y_range[1], zz, color="g")
    ax3d.plot_surface(xx, y_range[1], zz, color="g", alpha=0.2)

if PLOT_MODEL:
    # Add model to plot list
    envir.plot_structs.append(plot_model_rect)
    envir.plot_structs_args.append((model_bounds,))


##########              Plot!               ###########
s.plot_all('brine_shrimp_IBFE.mp4', fps=10)
#s.plot_all()
