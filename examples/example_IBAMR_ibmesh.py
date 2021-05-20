#! /usr/bin/env python3
''' Loads last file in IBAMR cylinder data for flow field AND loads the
vtk vertex data, creating an ibmesh out of it using convex hull. Agents should
not be able to move through the cylinder boundaries.'''

import numpy as np
import sys
sys.path.append('..')
import Planktos


envir = Planktos.environment()
envir.read_IBAMR3d_vtk_dataset('../tests/IBAMR_test_data', start=5, finish=None)
envir.read_vertex_data_to_convex_hull('../tests/IBAMR_test_data/mesh_db.vtk')

# tile flow in a 3,3 grid
# envir.tile_flow(3,3)

envir.add_swarm(init=(envir.L[0]*0.5,0.04,envir.L[2]*0.1))
s = envir.swarms[0]
# amount of jitter (variance)
s.shared_props['cov'] *= 0.0001

print('Moving swarm...')
for ii in range(40):
    s.move(0.1)

s.plot_all(movie_filename='IBAMR_cyl_ibmesh.mp4', fps=3)
