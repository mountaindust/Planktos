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
PLOT_CYL = False

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
print('-------------------------------------------')
# NOW:
# Domain should be 80x420x80 mm
# Model sits (from,to): (2.5,77.5), (85,235), (0.5,20.5)

# Add swarm right in front of model
s = envir.add_swarm(swarm_s=1000, init='point', pos=(40,84,1))

# Specify amount of jitter (mean, covariance)
# Set sigma**2 as 0.5cm**2/sec =  
# (sigma**2=2*D, D for brine shrimp given in Kohler, Swank, Haefner, Powell 2010)
shrimp_walk = ([0,0,0], 50*np.eye(3))

print('Moving swarm...')
for ii in range(240):
    s.move(0.1, shrimp_walk)

########## This bit is for cylinders. Change to represent model ##########

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
#s.plot_all()
