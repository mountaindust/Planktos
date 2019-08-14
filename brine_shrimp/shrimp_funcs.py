'''
Supporting functions for shrimp data visualization and persistance
'''

import sys
sys.path.append('..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

############################################################################
############           DATA GATHERING/SAVING FUNCTIONS          ############
############################################################################

def collect_cell_counts(swm, g_bounds, b_bounds, cell_size):
    '''Collect and return counts of shrimp in experiemental cells.
    
    Arguments:
        swm: swarm object
        g_bounds: bounds of green zone in y-direction
        b_bounds: bounds of blue zone in y-direction
        cell_size: width of each square cell in mm
        
    Returns:
        g_cells_cnts: For each time, cell counts in each green cell
        b_cells_cnts: For each time, cell counts in each blue cell'''

    gy_cells = [(g_bounds[0]+cell_size*k, g_bounds[0]+cell_size*k+cell_size) for k in np.arange(8)%2]
    by_cells = [(b_bounds[0]+cell_size*k, b_bounds[0]+cell_size*k+cell_size) for k in np.arange(8)%2]
    z_cells = [(cell_size*k, cell_size*k+cell_size) for k in np.arange(8)//2]

    # Tabulate counts for each cell
    print('Obtaining counts...')
    g_cells_cnts = list()
    b_cells_cnts = list()
    for shrimps in swm.pos_history: # each time point in history
        g_cells_cnts.append([])
        b_cells_cnts.append([])
        for gcell, zcell in zip(gy_cells, z_cells): # each green cell
            g_cells_cnts[-1].append(np.logical_and(
                (gcell[0] <= shrimps[:,1]) & (shrimps[:,1] < gcell[1]),
                (zcell[0] <= shrimps[:,2]) & (shrimps[:,2] < zcell[1])
            ).sum())
        for bcell, zcell in zip(by_cells, z_cells): # each blue cell
            b_cells_cnts[-1].append(np.logical_and(
                (bcell[0] <= shrimps[:,1]) & (shrimps[:,1] < bcell[1]),
                (zcell[0] <= shrimps[:,2]) & (shrimps[:,2] < zcell[1])
            ).sum())

    # append last (current) time point
    g_cells_cnts.append([])
    b_cells_cnts.append([])
    for gcell, zcell in zip(gy_cells, z_cells): # each green cell
        g_cells_cnts[-1].append(np.logical_and(
            (gcell[0] <= swm.positions[:,1]) & (swm.positions[:,1] < gcell[1]),
            (zcell[0] <= swm.positions[:,2]) & (swm.positions[:,2] < zcell[1])
        ).sum())
    for bcell, zcell in zip(by_cells, z_cells): # each blue cell
        b_cells_cnts[-1].append(np.logical_and(
            (bcell[0] <= swm.positions[:,1]) & (swm.positions[:,1] < bcell[1]),
            (zcell[0] <= swm.positions[:,2]) & (swm.positions[:,2] < zcell[1])
        ).sum())

    return g_cells_cnts, b_cells_cnts



def collect_zone_statistics(swm, g_bounds, b_bounds):
    '''Find the mean time (and std) for shrimp to enter each zone.

    Arguments:
        swm: swarm object
        g_bounds: bounds of green zone in y-direction
        b_bounds: bounds of blue zone in y-direction

    Returns:
        g_mean: mean entry time of green zone
        g_std: std of entry time of green zone
        b_mean: mean entry time of blue zone
        b_std: std of entry time of blue zone
    '''

    g_zone_crossings = []
    b_zone_crossings = []
    g_zone_counted = np.zeros(swm.positions.shape[0], dtype=bool)
    b_zone_counted = np.array(g_zone_counted)

    print('Obtaining zone statistics...')
    for shrimps in swm.pos_history:
        # count how many first crossings there are and record
        g_crossings = np.logical_and(shrimps[:,1] >= g_bounds[0], 
                                     np.logical_not(g_zone_counted))
        g_zone_crossings.append(g_crossings.sum())
        # mark the ones that crossed
        g_zone_counted[g_crossings] = True
        # repeat for blue zone
        b_crossings = np.logical_and(shrimps[:,1] >= b_bounds[0], 
                                     np.logical_not(b_zone_counted))
        b_zone_crossings.append(b_crossings.sum())
        b_zone_counted[b_crossings] = True
    # repeat for the current (final) time
    g_crossings = np.logical_and(swm.positions[:,1] >= g_bounds[0], 
                                    np.logical_not(g_zone_counted))
    g_zone_crossings.append(g_crossings.sum())
    g_zone_counted[g_crossings] = True
    b_crossings = np.logical_and(swm.positions[:,1] >= b_bounds[0], 
                                    np.logical_not(b_zone_counted))
    b_zone_crossings.append(b_crossings.sum())
    b_zone_counted[b_crossings] = True

    # report how many never crossed
    print('{} shrimp never entered the green zone.'.format(
        np.logical_not(g_zone_counted).sum()
    ))
    print('{} shrimp never entered the blue zone.'.format(
        np.logical_not(b_zone_counted).sum()
    ))

    # compute statistics
    all_time = np.array(swm.envir.time_history + [swm.envir.time])
    g_mean = np.average(all_time, weights=g_zone_crossings)
    g_std = np.sqrt(np.average((all_time - g_mean)**2, weights=g_zone_crossings))
    b_mean = np.average(all_time, weights=b_zone_crossings)
    b_std = np.sqrt(np.average((all_time - b_mean)**2, weights=b_zone_crossings))

    return g_mean, g_std, b_mean, b_std
    


def plot_cell_counts(time_mesh, g_cells_cnts, b_cells_cnts, prefix=''):
    '''Create plots of cell counts using all time points and save to pdf.
    Can add a prefix to the filenames.'''
    plot_order = [7,8,5,6,3,4,1,2]
    g_cells_cnts = np.array(g_cells_cnts)
    b_cells_cnts = np.array(b_cells_cnts)

    plt.figure(figsize=(4.8, 6.4))
    for n, plot in enumerate(plot_order):
        plt.subplot(4,2,plot)
        plt.plot(time_mesh, g_cells_cnts[:,n])
        plt.xlabel('time (s)')
        plt.ylabel('counts')
        plt.title('Cell number {}'.format(n+1))
    plt.tight_layout()
    plt.savefig(prefix+'green_cell_plots.pdf')

    plt.figure(figsize=(4.8, 6.4))
    for n, plot in enumerate(plot_order):
        plt.subplot(4,2,plot)
        plt.plot(time_mesh, b_cells_cnts[:,n])
        plt.xlabel('time (s)')
        plt.ylabel('counts')
        plt.title('Cell number {}'.format(n+1))
    plt.tight_layout()
    plt.savefig(prefix+'blue_cell_plots.pdf')



def save_sim_to_excel(time_mesh, g_cells_cnts, b_cells_cnts, prefix=''):
    '''Save simulation data to an excel spreadsheet using pandas.
    Can add a prefix to the filenames.'''
    # Experimental data was every 0.5 sec
    df_g = pd.DataFrame(g_cells_cnts[0::5], index=time_mesh[0::5], columns=list(np.arange(1,9)))
    df_g.to_excel(prefix+'green_cell_data.xlsx')
    df_b = pd.DataFrame(b_cells_cnts[0::5], index=time_mesh[0::5], columns=list(np.arange(1,9)))
    df_b.to_excel(prefix+'blue_cell_data.xlsx')


##########################################################################
############              MOVIE RELATED FUNCTIONS             ############
##########################################################################
# These functions to be added to swarm plot list to put structures in the movie

def plot_model_rect(ax3d, bounds):
    '''Plot the model as a translucent rectangular prism

    Arguments:
        ax3d: Axes3D object
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)'''
    x_range = bounds[0:2]
    y_range = bounds[2:4]
    z_range = bounds[4:]

    xx, yy = np.meshgrid(x_range, y_range)
    ax3d.plot_wireframe(xx, yy, z_range[0]*np.ones_like(xx), color="lightgray")
    ax3d.plot_surface(xx, yy, z_range[0]*np.ones_like(xx), color="lightgray", alpha=0.2)
    ax3d.plot_wireframe(xx, yy, z_range[1]*np.ones_like(xx), color="lightgray")
    ax3d.plot_surface(xx, yy, z_range[1]*np.ones_like(xx), color="lightgray", alpha=0.2)

    yy, zz = np.meshgrid(y_range, z_range)
    ax3d.plot_wireframe(x_range[0]*np.ones_like(yy), yy, zz, color="lightgray")
    ax3d.plot_surface(x_range[0]*np.ones_like(yy), yy, zz, color="lightgray", alpha=0.2)
    ax3d.plot_wireframe(x_range[1]*np.ones_like(yy), yy, zz, color="lightgray")
    ax3d.plot_surface(x_range[1]*np.ones_like(yy), yy, zz, color="lightgray", alpha=0.2)

    xx, zz = np.meshgrid(x_range, z_range)
    ax3d.plot_wireframe(xx, y_range[0]*np.ones_like(xx), zz, color="lightgray")
    ax3d.plot_surface(xx, y_range[0]*np.ones_like(xx), zz, color="lightgray", alpha=0.2)
    ax3d.plot_wireframe(xx, y_range[1]*np.ones_like(xx), zz, color="lightgray")
    ax3d.plot_surface(xx, y_range[1]*np.ones_like(xx), zz, color="lightgray", alpha=0.2)



def plot_sample_areas(ax3d, g_range, b_range, x_range):
    '''Plot the sample areas on the base of the domain

    Arguments:
        ax3d: Axes3D object
        g_bounds: (ymin, ymax)
        b_bounds: (ymin, ymax)
        x_bounds: (xmin, xmax)'''
    xx, yy = np.meshgrid(x_range, g_range)
    ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color="mediumseagreen")
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), color="mediumseagreen", alpha=0.2)
    
    xx, yy = np.meshgrid(x_range, b_range)
    ax3d.plot_wireframe(xx, yy, np.zeros_like(xx), color="cornflowerblue")
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), color="cornflowerblue", alpha=0.2)
