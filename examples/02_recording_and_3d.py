#! /usr/bin/env python3

'''
This is a tutorial and minimal working example in 3D which builds on the 2D
example and further illustrates the main points of creating a simulation in
Planktos. It also shows how to record a simulation to disk as it runs, and how
to read the recording back or pick the simulation up again where it left off.
'''

import planktos

# To create a 3D environment in Planktos, all you need to do is specify the
#   length of the third (z) dimension. Let's go ahead and specify the length
#   of all three dimensions this time (in meters). We'll plan on using Brinkman
#   flow again, so let's also go ahead and specify the fluid density and
#   dynamic viscosity:
envir = planktos.Environment(Lx=20, Ly=10, Lz=4, rho=1000, mu=1000)

# Now let's set the Brinkman flow. It automatically gets translated into all
#   available spatial dimensions:
envir.set_brinkman_flow(alpha=66, h_p=1.5, U=1, dpdx=1, res=101)

# If you decide later you want a bigger spatial domain (or you loaded fluid data
#   with periodic boundary conditions but want a bigger domain than the CFD
#   data was generated on), you can tile the environment along any axis using
#   the tile_domain method, e.g. envir.tile_domain(x=2, y=3, z=1) -- z exists in
#   3D only, and this also works in 2D. The fluid data is not copied, and the
#   immersed mesh is copied into every tile unless you pass tile_ibmesh=False.
#   Tiling must be the last change you make to the environment, and it must
#   come before the simulation starts: after loading the fluid and the mesh, and 
#   before creating swarms.

# 3D visualization is a hard problem in general and requires advanced libraries
#   specifically built for that purpose (e.g., VTK). While Planktos can read
#   VTK files, 3D visualization of data is not the main purpose of Planktos and
#   so functionality is currently limited. This may be improved in the future
#   by replacing matplotlib-based visualization with VTK-based visualization for
#   3D problems, but for now, you should not expect the aspect ratio for 3D
#   environments to be correct, and some features (such as vorticity plotting)
#   are only available in 2D.
envir.plot_flow()

# Now let's repeat the first steps from 01_basic_2d.py, creating a default
#   Swarm and changing the covariance.

swrm = planktos.Swarm(envir=envir, seed=1)
swrm.shared_props['cov'] = swrm.shared_props['cov'] * 0.01

# This time, let's record the simulation as it runs. Environment.record writes
#   the state of every swarm in the environment to a directory on disk, a chunk
#   at a time, so the data survives a crash and a long run is not limited by
#   memory. Put the simulation loop inside a with block: recording starts when
#   the block opens and stops when it closes.
#
# What gets recorded:
#   - store= chooses which agent data are kept at each time step. The default is
#     positions only; velocities, accelerations, props and shared_props can be
#     added, e.g. store=('positions', 'velocities').
#   - fluid= chooses which fluid backdrops (vorticity, quiver arrows) a later
#     plot_all can draw straight from the recording. These backdrops exist only
#     in 2D, so in this 3D simulation fluid= is ignored and the recording holds
#     agent data only. In 2D, recording them means you can make a movie of a
#     large fluid dataset without reading that dataset from disk a second time.
#   - capture_interval=k keeps only every k-th time step, both on disk and in
#     the swarm's history in memory.
#
# If the directory you give already exists and is not empty (for example,
#   because you have run this script before), the simulation is recorded to a
#   new, timestamped directory beside it instead, with a warning. rec.path
#   always says where the data went.
#
# plot_all= makes a movie of the run when the with block ends; its entries are
#   passed on to Swarm.plot_all. playback_rate=5 plays five simulated seconds
#   per second of video (e.g. a 5x speed video), so at plot_all's default of 10 
#   frames per second, every 5th time step (dt=0.1) gets plotted. (plot_all also 
#   accepts frames=, a list of the time step indices to plot.) Whenever a 
#   recorded run is plotted, here or by calling plot_all later, any fluid 
#   backdrop comes from the recording, which has pre-computed and downsampled 
#   quantities, rather than from the fluid data itself which could be large.

print('Moving swarm...')
with envir.record('results/basic_3d_run',
                  plot_all=dict(playback_rate=5)) as rec:
    for ii in range(240):
        swrm.move(0.1)

# planktos.load_run opens a recording, in this session or any later one,
#   without re-running anything. Swarms are numbered from 0 in the order they
#   were added to the environment. Data is read from disk only when you ask for
#   it, so a recording larger than memory is still usable. Using a with block
#   closes the files afterward (on Windows, a recording cannot be moved or
#   deleted while it is open).
with planktos.load_run(rec.path) as run:
    print(run)  # number of recorded time steps and swarms, and the time span

    # positions has one entry per recorded time step. capture_at finds the one
    #   nearest a given time, and each entry is a masked array, just like
    #   Swarm.positions: masked rows are agents that have left the domain.
    positions = run.positions(0) # there's only one swarm, swarm 0
    pos_12 = positions[run.capture_at(12)] # the positions of all agents nearest t=12
    print('Mean agent position at t=12: {}'.format(pos_12.mean(axis=0).round(2)))

    # restore rebuilds the Environment and its Swarms exactly as the run left
    #   them -- positions, properties, random number stream and clock -- so you
    #   can continue the simulation or plot it in a new session or after a
    #   crash. The fluid and any immersed mesh are not saved in the recording;
    #   instead, it notes how they were loaded or generated, and restore repeats
    #   those steps. restore(capture=j) rebuilds the run at an earlier recorded
    #   time step j instead of at the end.
    envir2, swarms2 = run.restore()

swrm2 = swarms2[0] # get swarm 0

# Recording a restored run into the directory it came from adds to that
#   recording, so it ends up as if the run had never stopped.
print('Continuing the restored run...')
with envir2.record(rec.path):
    for ii in range(60):
        swrm2.move(0.1)

with planktos.load_run(rec.path) as run:
    print(run)
