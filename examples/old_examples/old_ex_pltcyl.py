#! /usr/bin/env python3
'''
This is a rather complicated example which shows how we can add objects to 
the swarm plots which aren't really there, at least in the sense that agents 
don't interact with them as they do loaded vertex meshes. It's mostly here for
record-keeping purposes and you should skip it unless this is something you
very specifically want to do for some reason.
'''

import sys
sys.path.append('../..')
import numpy as np
import planktos

def add_cylinders_toplot(envir, filename):
    '''Load unstructred grid points (vtk data) for a cylinder and add it to
    the envir object to be plotted as a surface. If the environment has been
    tiled, automatically tile the cylinder too.'''

    points, bounds = planktos.dataio.read_vtk_Unstructured_Grid_Points(filename)
    # shift to first quadrant
    for dim in range(3):
        bounds[dim*2:dim*2+2] -= envir.fluid_domain_LLC[dim]

    if envir.tiling is not None:
        # add a cylinder for each tile
        for ii in range(envir.tiling[0]):
            for jj in range(envir.tiling[1]):
                envir.plot_structs.append(_plot_cylinders)
                these_bounds = np.array(bounds)
                these_bounds[:2] += envir.orig_L[0]*ii
                these_bounds[2:4] += envir.orig_L[1]*jj
                envir.plot_structs_args.append((these_bounds,))
    else:
        # add a cylinder to be plotted
        envir.plot_structs.append(_plot_cylinders)
        envir.plot_structs_args.append((bounds,))

def _plot_cylinders(ax3d, bounds):
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

envir = planktos.Environment()
envir.read_IBAMR3d_vtk_dataset('../tests/IBAMR_test_data', start=5, finish=None)
# tile flow in a 3,3 grid
envir.tile_flow(3,3)

# plot cylinder in each tile
add_cylinders_toplot(envir, '../tests/IBAMR_test_data/mesh_db.vtk')

envir.add_swarm()
s = envir.swarms[0]
# amount of jitter (variance)
s.shared_props['cov'] *= 0.0001

print('Moving swarm...')
for ii in range(50):
    s.move(0.1)

s.plot_all()
