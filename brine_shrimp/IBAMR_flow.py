#! /usr/bin/env python3

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import framework, data_IO

# Whether or not to show the cylinders based on the mesh data
PLOT_CYL = True

# Intialize environment
envir = framework.environment()

# Import IBMAR data on flow
envir.read_IBAMR3d_vtk_data('data/16towers_Re10_len10.vtk')
print('Domain set to {}.'.format(envir.L))

envir.add_swarm()
s = envir.swarms[0]

# Specify amount of jitter (mean, covariance)
# Set std as 1 cm = 0.01 m
shrimp_walk = ([0,0,0], (0.01**2)*np.eye(3))

print('Moving swarm...')
for ii in range(240):
    s.move(0.1, shrimp_walk)

########## This bit is for plotting the cylinders, if so desired ##########

def plot_cylinders(ax3d, bounds):
    '''Plot a vertical cylinder on a matplotlib Axes3D object.
    
    Arguments:
        ax3d: Axes3D object
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)'''

    # Make data for plot_surface
    theta = np.linspace(0, 2 * np.pi, 11)
    height = np.linspace(bounds[4], bounds[5], 11)
    r = (bounds[1] - bounds[0])/2
    center = (bounds[0]+r,bounds[2]+(bounds[3]-bounds[2])/2)

    x = r * np.outer(np.cos(theta), np.ones(np.size(height)))+center[0]
    y = r * np.outer(np.sin(theta), np.ones(np.size(height)))+center[1]
    z = np.outer(np.ones(np.size(theta)), height)

    ax3d.plot_surface(x, y, z, color='g', alpha=0.3)

if PLOT_CYL:
    # get mesh data and translate to new domain
    N_cyl = 16 # number of cylinder files
    for n in range(N_cyl):
        if n<10:
            n_str = '0'+str(n)
        else:
            n_str = str(n)
        points, bounds = data_IO.read_vtk_Unstructured_Grid_Points(
                        'data/cyl_grids_'+n_str+'.vtk')
        # shift to first quadrant
        for dim in range(3):
            bounds[dim*2:dim*2+2] -= envir.fluid_domain_LLC[dim]
        # add a cylinder plot
        envir.plot_structs.append(plot_cylinders)
        envir.plot_structs_args.append((bounds,))

##########              Plot!               ###########
s.plot_all('brine_shrimp_IBAMR.mp4', fps=20)
# s.plot_all()
