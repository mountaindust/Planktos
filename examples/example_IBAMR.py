#! /usr/bin/env python3
''' Loads last file in IBAMR test data and moves a swarm in the flow'''

import numpy as np
import sys
sys.path.append('..')
import agents, data_IO

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

    ax3d.plot_surface(x, y, z, color='g')



envir = agents.environment()
envir.read_IBAMR3d_vtk_data('../tests/IBAMR_test_data', start=5, finish=None)
# tile flow in a 3,3 grid
envir.tile_flow(3,3)
# get mesh data and translate to new domain
points, bounds = data_IO.read_vtk_Unstructured_Grid_Points(
                 '../tests/IBAMR_test_data/mesh_db.vtk')
for dim in range(3):
    bounds[dim*2:dim*2+2] -= envir.fluid_domain_LLC[dim]
# add a cylinder plot for each tile
for ii in range(envir.tiling[0]):
    for jj in range(envir.tiling[1]):
        envir.plot_structs.append(plot_cylinders)
        these_bounds = np.array(bounds)
        these_bounds[:2] += envir.orig_L[0]*ii
        these_bounds[2:4] += envir.orig_L[1]*jj
        envir.plot_structs_args.append((these_bounds,))

envir.add_swarm()
s = envir.swarms[0]

print('Moving swarm...')
for ii in range(50):
    s.move(0.1, params=(np.zeros(3), np.eye(3)*0.0001))

s.plot_all()
