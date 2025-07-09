'''
Supporting functions for shrimp data visualization and persistance
'''

import sys
sys.path.append('../..')
from sys import platform
if platform == 'darwin': # OSX backend does not support blitting
    import matplotlib
    matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

############################################################################
############           DATA GATHERING/SAVING FUNCTIONS          ############
############################################################################

def collect_cell_counts(swm, g_bounds, b_bounds, cell_size):
    '''Collect and return counts of shrimp in experiemental cells.
    
    Arguments:
        swm: Swarm object
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
        # shrimps is a masked array
        g_cells_cnts.append([])
        b_cells_cnts.append([])
        for gcell, zcell in zip(gy_cells, z_cells): # each green cell
            g_cells_cnts[-1].append(np.logical_and(
                (gcell[0] <= shrimps[:,1].data) & (shrimps[:,1].data < gcell[1]),
                (zcell[0] <= shrimps[:,2].data) & (shrimps[:,2].data < zcell[1])
            ).sum())
        for bcell, zcell in zip(by_cells, z_cells): # each blue cell
            b_cells_cnts[-1].append(np.logical_and(
                (bcell[0] <= shrimps[:,1].data) & (shrimps[:,1].data < bcell[1]),
                (zcell[0] <= shrimps[:,2].data) & (shrimps[:,2].data < zcell[1])
            ).sum())

    # append last (current) time point
    g_cells_cnts.append([])
    b_cells_cnts.append([])
    for gcell, zcell in zip(gy_cells, z_cells): # each green cell
        g_cells_cnts[-1].append(np.logical_and(
            (gcell[0] <= swm.positions[:,1].data) & (swm.positions[:,1].data < gcell[1]),
            (zcell[0] <= swm.positions[:,2].data) & (swm.positions[:,2].data < zcell[1])
        ).sum())
    for bcell, zcell in zip(by_cells, z_cells): # each blue cell
        b_cells_cnts[-1].append(np.logical_and(
            (bcell[0] <= swm.positions[:,1].data) & (swm.positions[:,1].data < bcell[1]),
            (zcell[0] <= swm.positions[:,2].data) & (swm.positions[:,2].data < zcell[1])
        ).sum())

    return g_cells_cnts, b_cells_cnts



def collect_zone_statistics(swm, g_bounds, b_bounds):
    '''Find the mean time (and std) for shrimp to enter each zone.

    Arguments:
        swm: Swarm object
        g_bounds: bounds of green zone in y-direction
        b_bounds: bounds of blue zone in y-direction

    Returns:
        g_cross_frac: fraction crossing into green zone during sim
        g_mean: mean entry time of green zone (given entry happened)
        g_std: std of entry time of green zone
        g_skew: Pearson's moment coefficient of skewness for green zone entry
        g_kurt: Pearson's moment coefficient of kurtosis for green zone entry
        b_cross_frac: fraction crossing into blue zone during sim
        b_mean: mean entry time of blue zone (given entry happened)
        b_std: std of entry time of blue zone
        b_skew: Pearson's moment coefficient of skewness for blue zone entry
        b_kurt: Pearson's moment coefficient of kurtosis for blue zone entry
    '''

    g_zone_crossings = []
    b_zone_crossings = []
    g_zone_counted = np.zeros(swm.positions.shape[0], dtype=bool)
    b_zone_counted = np.array(g_zone_counted)

    print('Obtaining zone statistics...')
    for shrimps in swm.pos_history:
        # count how many first crossings there are and record
        #   careful!! position is a masked array!
        g_crossings = np.logical_and(shrimps[:,1].data >= g_bounds[0], 
                                     np.logical_not(g_zone_counted))
        g_zone_crossings.append(g_crossings.sum())
        # mark the ones that crossed
        g_zone_counted[g_crossings] = True
        # repeat for blue zone
        b_crossings = np.logical_and(shrimps[:,1].data >= b_bounds[0], 
                                     np.logical_not(b_zone_counted))
        b_zone_crossings.append(b_crossings.sum())
        b_zone_counted[b_crossings] = True
    # repeat for the current (final) time
    g_crossings = np.logical_and(swm.positions[:,1].data >= g_bounds[0], 
                                    np.logical_not(g_zone_counted))
    g_zone_crossings.append(g_crossings.sum())
    g_zone_counted[g_crossings] = True
    b_crossings = np.logical_and(swm.positions[:,1].data >= b_bounds[0], 
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
    g_cross_frac = g_zone_counted.sum()/len(g_zone_counted)
    b_cross_frac = b_zone_counted.sum()/len(b_zone_counted)

    # compute statistics
    # as a weighted time avg, this is an expected value conditioned on crossing
    # skew and kurtosis are Pearson coefficients
    g_zone_crossings = np.array(g_zone_crossings)
    b_zone_crossings = np.array(b_zone_crossings)
    # sanity check
    try:
        assert g_zone_counted.sum() == g_zone_crossings.sum()
        assert b_zone_counted.sum() == b_zone_crossings.sum()
    except AssertionError:
        import pdb; pdb.set_trace()
    #
    all_time = np.array(swm.envir.time_history + [swm.envir.time])
    g_mean = np.average(all_time, weights=g_zone_crossings)
    g_std = np.sqrt(np.average((all_time - g_mean)**2, weights=g_zone_crossings))
    if g_std != 0:
        g_skew = np.average(((all_time - g_mean)/g_std)**3, weights=g_zone_crossings)
        g_kurt = np.average(((all_time - g_mean)/g_std)**4, weights=g_zone_crossings)
    else:
        g_skew = 0.0
        g_kurt = 0.0
    b_mean = np.average(all_time, weights=b_zone_crossings)
    b_std = np.sqrt(np.average((all_time - b_mean)**2, weights=b_zone_crossings))
    if b_std != 0:
        b_skew = np.average(((all_time - b_mean)/b_std)**3, weights=b_zone_crossings)
        b_kurt = np.average(((all_time - b_mean)/b_std)**4, weights=b_zone_crossings)
    else:
        b_skew = 0.0
        b_kurt = 0.0

    # Get the median
    g_num = g_zone_crossings.sum()
    g_mid = int(g_num/2)
    b_num = b_zone_crossings.sum()
    b_mid = int(b_num/2)
    for n, val in enumerate(np.cumsum(g_zone_crossings)):
        if val >= g_mid:
            if g_num % 2 == 0 and val == g_mid:
                # even and necessary to get adverage
                g_median = (all_time[n] + all_time[n+1])/2
                break
            else:
                g_median = all_time[n]
                break
    for n, val in enumerate(np.cumsum(b_zone_crossings)):
        if val >= b_mid:
            if b_num % 2 == 0 and val == b_mid:
                # even and necessary to get adverage
                b_median = (all_time[n] + all_time[n+1])/2
                break
            else:
                b_median = all_time[n]
                break
    # Get the mode
    g_mode = all_time[np.argmax(g_zone_crossings)]
    b_mode = all_time[np.argmax(b_zone_crossings)]

    return g_cross_frac, g_mean, g_median, g_mode, g_std, g_skew, g_kurt,\
        b_cross_frac, b_mean, b_median, b_mode, b_std, b_skew, b_kurt
    


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
