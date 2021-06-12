#! /usr/bin/env python3
'''
This example is very similar to ex_ib2d_ibmesh.py, but in 3D and with data 
that is available without external download. It also demonstrates the utility of 
the convex hull algorithm and demonstrates how the domain can be tiled.

This example loads the vtk fluid velocity field AND loads vertex data describing 
a 3D cylinder, creating an ibmesh out of it using a convex hull algorithm. \
Agents should not be able to move through the cylinder boundaries.'''

import sys
sys.path.append('..')
import planktos

# Let's begin by creating a default environment.
envir = planktos.environment()

# Now, load the VTK data. This is just an excerpt from the larger data set, and
#   only file dumps 3-5 are included. So we will start at 3 and go until there
#   are none left. This is just a bit of data which originally came from IBAMR 
#   that we use for testing purposes, thus it's location in the tests folder.
envir.read_IBAMR3d_vtk_dataset('../tests/IBAMR_test_data', start=3, finish=None)

# Now we read in the vertex data. Unlike the ib2d_ibmesh example, here we will
#   use a convex hull algorithm to create a solid object out of the vertex 
#   points. This way, we don't end up with a cylinder without a top on it!
envir.read_vertex_data_to_convex_hull('../tests/IBAMR_test_data/mesh_db.vtk')
# Plot to see what we have:
envir.plot_flow()

# This data was generated with periodic boundary conditions in the x and y
#   directions, simulating an infinite array of cylinders. This means that we
#   can tile our environment if we like, getting a bigger domain for the agents!
#   In this case, let's tile the environment to a 3x3 grid with respect to its
#   original size.
envir.tile_flow(3,3)

# Now we can see the effect - notice that the cylinder mesh got tiled too!!
envir.plot_flow()

# Let's add a swarm with 100 agents all positioned somewhat behind the 
#   center left-most cylinder with respect to the flow (which is in the 
#   y-direction in this example). Remember that we can do this by specifying a 
#   point to the init argument of the swarm class, and that we can get the 
#   length of the domain in each direction with the envir.L attribute. 
swrm = planktos.swarm(envir=envir, init=(envir.L[0]*0.5, 0.04, envir.L[2]*0.1))

# adjust the amount of jitter (variance)...
swrm.shared_props['cov'] *= 0.0001

# ...and now move the swarm.
print('Moving swarm...')
for ii in range(40):
    swrm.move(0.1)

swrm.plot_all()
