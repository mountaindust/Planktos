#! /usr/bin/env python3
'''
Read COMSOL data for 3D cylinder model. Reproduce flowtank experiment with
random walk agents.
'''

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import argparse
import numpy as np
import shrimp_funcs
import Planktos, data_IO

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000,
                    help="number of shrimp to simulate")
parser.add_argument("-s", "--seed", type=int, default=1,
                    help="seed for the random number generator")
parser.add_argument("-o", "--prefix", type=str, 
                    help="prefix to filenames with data output",
                    default='')
parser.add_argument("-d", "--data", type=str, 
                    help="name of flow data file within ./data",
                    default='')
parser.add_argument("--movie", action="store_true", default=False,
                    help="output a movie of the simulation")
parser.add_argument("-t", "--time", type=int, default=55,
                    help="time in sec to run the simulation")


############################################################################
############                    RUN SIMULATION                  ############
############################################################################

def main(swarm_size=1000, time=600, data='', seed=1, create_movie=False, prefix=''):
    '''Add swarm and simulate dispersal. Future: run this in loop while altering
    something to see the effect.'''

    # Intialize environment
    envir = Planktos.environment(x_bndry=['noflux', 'noflux'])

    ############     Import COMSOL data on flow     ############

    print('Reading COMSOL data. This will take a while...')
    if data == '':
        envir.read_comsol_vtu_data('data/Velocity_plate.vtu', vel_conv=1000)
    else:
        if data[-4:] != '.txt' and data[-4:] != '.vtu':
            data = data+'.vtu'
        try:
            envir.read_comsol_vtu_data(data, vel_conv=1000)
        except AssertionError:
            envir.read_comsol_vtu_data('data/'+data, vel_conv=1000)
    print('Domain set to {} mm.'.format(envir.L))
    print('Flow mesh is {}.'.format(envir.flow[0].shape))
    # Domain should be 80x640x80 mm
    # Model sits (from,to): (2.5,77.5), (85,235), (0.5,20.5)
    model_bounds = (2.5,77.5,85,235,0,20.5)
    print('-------------------------------------------')

    # Add swarm right in front of model
    s = envir.add_swarm(swarm_s=swarm_size, init='point', pos=(40,84,3), seed=seed)

    # Specify amount of jitter (mean, covariance)
    # (sigma**2=2*D, D for brine shrimp given in Kohler, Swank, Haefner, Powell 2010)
    # D was found to be 0.025 cm**2/sec (video data, real time was higher)
    # Set sigma**2 as 2*D = 0.05cm**2/sec = 5mm**2/sec, sigma~7mm
    
    shrimp_walk = ([0,0,0], 5*np.eye(3))

    ########## Move the swarm according to the prescribed rules above ##########
    print('Moving swarm...')
    dt = 0.1
    num_of_steps = time*10
    for ii in range(num_of_steps):
        s.move(dt, shrimp_walk)


    ############## Gather data about flow tank observation area ##############
    # Observation squares are 2cm**2 until the top
    # Green zone begins right where the model ends, at y=235. Ends at y=275 mm.
    g_bounds = (235, 275)
    # Blue zone begins 14.2-4=10.2 cm later, at y=377 mm. Ends at y=417 mm.
    b_bounds = (377, 417)
    # Each cell is 2cm x 2cm
    cell_size=20

    g_cells_cnts, b_cells_cnts = shrimp_funcs.collect_cell_counts(s, g_bounds,
                                                            b_bounds, cell_size)

    g_mean, g_std, b_mean, b_std = shrimp_funcs.collect_zone_statistics(s, g_bounds, b_bounds)


    ########## Plot and save the run ##########
    # Create time mesh for plotting and saving data in excel
    time_mesh = list(envir.time_history)
    time_mesh.append(envir.time)
    # Plot and save the run
    shrimp_funcs.plot_cell_counts(time_mesh, g_cells_cnts, b_cells_cnts, prefix)
    shrimp_funcs.save_sim_to_excel(time_mesh, g_cells_cnts, b_cells_cnts, prefix)
    # Output the zone stats
    np.savez(prefix+'_stats', g_mean=g_mean, g_std=g_std, b_mean=b_mean, b_std=b_std)

    
    ########## Create movie if requested ##########
    # This takes a long time. Only do it if asked to.
    if create_movie:
        # Add model to plot list
        envir.plot_structs.append(shrimp_funcs.plot_model_rect)
        envir.plot_structs_args.append((model_bounds,))
        # Add sample areas to plot list
        envir.plot_structs.append(shrimp_funcs.plot_sample_areas)
        envir.plot_structs_args.append((g_bounds, b_bounds, (0, envir.L[0])))
        # Call plot_all to create movie
        print('Creating movie...')
        s.plot_all(prefix+'brineshr_COMSOL.mp4', fps=10)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.prefix != '':
        prefix = args.prefix + '_'
    else:
        prefix = args.prefix
    main(args.N, args.time, args.data, args.seed, args.movie, prefix)

