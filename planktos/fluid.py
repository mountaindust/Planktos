'''Functions and methods for loading and handling fluid data.
These are mainly utilized by the Enviornment class.

Created: Thurs July 9 2025

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import numpy as np
from . import dataio

def read_IB2d_dumpfiles(path, d_start, d_finish, vector_data):
    '''
    Load IB2d data at path starting with dump d_start and ending with dump
    d_end. This can read just one or many dump files.

    Parameters
    ----------
    path : string
        path to vtk data
    d_start : int
        first dump file to load
    d_finish : int
        last dump file to load
    vector_data : bool
        whether the data is vector velocity data (u) or not. The other 
        choice being x and y directed velocity magnitude (uX, uY)

    Returns
    -------
    fluid : list of ndarray
    x : x-coordinate mesh, 1D ndarray
    y : y-coordinate mesh, 1D ndarray
    '''
    X_vel = []
    Y_vel = []
    for n in range(d_start, d_finish+1):
        # Points to desired data viz_IB2d data file
        if n < 10:
            numSim = '000'+str(n)
        elif n < 100:
            numSim = '00'+str(n)
        elif n < 1000:
            numSim = '0'+str(n)
        else:
            numSim = str(n)

        # Imports (x,y) grid values and ALL Eulerian Data %
        #                      DEFINITIONS
        #          x: x-grid                y: y-grid
        #       Omega: vorticity           P: pressure
        #    uMag: mag. of velocity
        #    uX: mag. of x-Velocity   uY: mag. of y-Velocity
        #    u: velocity vector
        #    Fx: x-directed Force     Fy: y-directed Force
        #
        #  Note: U(j,i): j-corresponds to y-index, i to the x-index
        
        if vector_data:
            # read in vector velocity data
            strChoice = 'u'; xy = True
            uX, uY, x, y = dataio.read_2DEulerian_Data_From_vtk(path, numSim,
                                                                strChoice,xy)
            X_vel.append(uX.T) # (y,x) -> (x,y) coordinates
            Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates
        else:
            # read in x-directed Velocity Magnitude #
            strChoice = 'uX'; xy = True
            uX,x,y = dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                        strChoice,xy)
            X_vel.append(uX.T) # (y,x) -> (x,y) coordinates

            # read in y-directed Velocity Magnitude #
            strChoice = 'uY'
            uY = dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                    strChoice)
            Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates

        ##### The following is just for reference! ######
        # read in Vorticity #
        # strChoice = 'Omega'; first = 0
        # Omega = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
        #                                               strChoice,first)
        # read in Pressure #
        # strChoice = 'P'; first = 0
        # P = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
        #                                           strChoice,first)
        # read in Velocity Magnitude #
        # strChoice = 'uMag'; first = 0
        # uMag = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
        #                                              strChoice,first)
        # read in x-directed Forces #
        # strChoice = 'Fx'; first = 0
        # Fx = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
        #                                            strChoice,first)
        # read in y-directed Forces #
        # strChoice = 'Fy'; first = 0
        # Fy = dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
        #                                            strChoice,first)
        ###################################################

    ### Return data ###
    if d_start != d_finish:
        return [np.transpose(np.dstack(X_vel),(2,0,1)), 
                np.transpose(np.dstack(Y_vel),(2,0,1))] , x, y
    else:
        return [X_vel[0], Y_vel[0]], x, y



def wrap_flow(flow, flow_points, periodic_dim=(True, True, False)):
    '''In some cases, software may print out fluid velocity data that omits 
    the velocities at the right boundaries in spatial dimensions that are 
    meant to be periodic. This helper function restores that data by copying 
    everything over. 3rd dimension will automatically be ignored if 2D.

    Parameters
    ----------
    flow : list of ndarrays
        This will be overwritten to save space!
    flow_points : tuple of mesh coordinates (x,y,[z])
    periodic_dim : list of 3 bool, default=[True, True, False]
        True if that spatial dimension is periodic, otherwise False

    Returns
    -------
    flow : list of ndarrays
    flow_points : tuple of mesh coordinates (ndarrays)
    L : list of dimension lengths
    '''

    dim = len(flow_points)
    if dim == len(flow[0].shape):
        TIME_DEP = False
    else:
        TIME_DEP = True
            
    dx = np.array([flow_points[d][-1]-flow_points[d][-2] 
                    for d in range(dim)])
    
    # find new flow field shape
    new_flow_shape = np.array(flow[0].shape)
    if not TIME_DEP:
        new_flow_shape += 1*np.array(periodic_dim)
    else:
        new_flow_shape[1:] += 1*np.array(periodic_dim)

    # create new flow field, putting old data in lower left corner
    new_flow = [np.zeros(new_flow_shape) for d in range(dim)]
    if TIME_DEP:
        old_shape = flow[0].shape[1:]
    else:
        old_shape = flow[0].shape
    for d in range(dim):
        if dim == 2:
            new_flow[d][...,:old_shape[0],:old_shape[1]] = flow[d]
        else:
            new_flow[d][...,:old_shape[0],:old_shape[1],:old_shape[2]] = flow[d]
    # replace old flow field
    flow = new_flow

    # fill in the new edges and update flow points
    flow_points_new = []
    for d in range(dim):
        if periodic_dim[d]:
            flow_points_new.append(np.append(flow_points[d], 
                                flow_points[d][-1]+dx[d]))
            for dd in range(dim):
                if d == 0 and not TIME_DEP:
                    flow[dd][-1,...] = flow[dd][0,...]
                elif d == 0 and TIME_DEP:
                    flow[dd][:,-1,...] = flow[dd][:,0,...]
                elif d == 1 and not TIME_DEP:
                    flow[dd][:,-1,...] = flow[dd][:,0,...]
                elif d == 1 and TIME_DEP:
                    flow[dd][:,:,-1,...] = flow[dd][:,:,0,...]
                else:
                    flow[dd][...,-1] = flow[dd][...,0]
        else:
            flow_points_new.append(flow_points[d])

    flow_pts = tuple(flow_points_new)
    # return flow, flow_points, L
    return flow, flow_pts, [flow_pts[d][-1] for d in range(dim)]


