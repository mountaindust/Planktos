'''Functions and methods for loading and handling fluid data.
These are mainly utilized by the Enviornment class.

Created: Thurs July 9 2025

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import warnings
import numpy as np
from scipy import interpolate
from pathlib import Path
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
    periodic_dim : list of 2 or 3 bool, default=[True, True, False]
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



def read_IBAMR3d_vtkfiles(path, d_start=0, d_finish=None, 
                          vel_conv=None, grid_conv=None):
    '''Reads in one or more vtk Rectilinear Grid Vector files. If path
    refers to a single file, the resulting flow will be time invarient.
    Otherwise, this method will assume that files are named IBAMR_db_###.vtk 
    where ### is the dump number, and that the mesh is the same in each vtk.
    Also, imported times will be translated backward so that the first time 
    loaded corresponds to a Planktos environment time of 0.0.

    Parameters
    ----------
    path : string
        path to vtk data, incl. file extension if a single file
    d_start : int, default=0
        vtk dump number to start with.
    d_finish : int, optional
        vtk dump number to end with. If None, end with last one.
    vel_conv : float, optional
        scalar to multiply the velocity by in order to convert units
    grid_conv : float, optional
        scalar to multiply the grid by in order to convert units

    Returns
    -------
    flow : list of ndarray (fluid data)
    mesh : list of 1D arrays of grid points in x, y, and z directions
    flow_times : None or ndarray of times at which the fluid velocity is
        specified.
    '''

    path = Path(path)
    if path.is_file():
        flow, mesh, time = dataio.read_vtk_Rectilinear_Grid_Vector(path)
        flow_times = None
    
    elif path.is_dir():
        file_names = [x.name for x in path.iterdir() if x.is_file() and
                      x.name[:9] == 'IBAMR_db_']
        file_nums = sorted([int(f[9:12]) for f in file_names])
        if d_start is None:
            d_start = file_nums[0]
        else:
            d_start = int(d_start)
            assert d_start in file_nums, "d_start number not found!"
        if d_finish is None:
            d_finish = file_nums[-1]
        else:
            d_finish = int(d_finish)
            assert d_finish in file_nums, "d_finish number not found!"

        ### Gather data ###
        flow = [[], [], []]
        flow_times = []

        for n in range(d_start, d_finish+1):
            if n < 10:
                num = '00'+str(n)
            elif n < 100:
                num = '0'+str(n)
            else:
                num = str(n)
            this_file = path / ('IBAMR_db_'+num+'.vtk')
            data, mesh, time = dataio.read_vtk_Rectilinear_Grid_Vector(str(this_file))
            for dim in range(3):
                flow[dim].append(data[dim])
            flow_times.append(time)

        flow = [np.array(flow[0]).squeeze(), np.array(flow[1]).squeeze(),
                np.array(flow[2]).squeeze()]
        # parse time information
        if None not in flow_times and len(flow_times) > 1:
            # shift time so that the first time is 0.
            flow_times = np.array(flow_times) - min(flow_times)
        elif None in flow_times and len(flow_times) > 1:
            # could not parse time information
            warnings.warn("Could not retrieve time information from at least"+
                          " one vtk file. Assuming unit time-steps...", UserWarning)
            flow_times = np.arange(len(flow_times))
        else:
            flow_times = None

    if vel_conv is not None:
        print("Converting vel units by a factor of {}.".format(vel_conv))
        for ii, d in enumerate(flow):
            flow[ii] = d*vel_conv
    if grid_conv is not None:
        print("Converting grid units by a factor of {}.".format(grid_conv))
        for ii, m in enumerate(mesh):
            mesh[ii] = m*grid_conv

    return flow, mesh, flow_times



#######################################################################
#####           FLUID TEMPORAL INTERPOLATION ROUTINES             #####
#######################################################################

def create_temporal_interpolations(flow_times, flow_data):
    '''This function controls how temporal interpolations of the fluid
    velocity data will be created for the Environment class.

    Parameters
    ----------
    flow_times : 1D array of time points
    flow_data : list of ndarrays with fluid velocity field data.
        This data should be expected to be overwritten to save space.

    Returns
    -------
    list of interpolation objects (PPoly)
    '''

    for n, flow in enumerate(flow_data):
        # Defaults to axis=0 along which data is varying, which is t axis
        # Defaults to not-a-knot boundary condition, resulting in first
        #   and second segments at curve ends being the same polynomial
        # Defaults to extrapolating out-of-bounds points based on first
        #   and last intervals. This will be overriden by this method
        #   to use constant extrapolation instead.
        flow_data[n] = fCubicSpline(flow_times, flow)
    return flow_data



class fCubicSpline(interpolate.CubicSpline):
    '''
    Extends Scipy's CubicSpline object to get info about original fluid data.
    '''

    def __init__(self, flow_times, flow, **kwargs):
        super(fCubicSpline, self).__init__(flow_times, flow, **kwargs)

        self.shape = flow.shape
        self.data_max = flow.max()
        self.data_min = flow.min()



    def __getitem__(self, pos):
        '''
        Allows indexing into the interpolator.
        '''
        if type(pos) == int:
            return self.__call__(self.x[pos])
        elif type(pos) == slice:
            start = pos.start; stop = pos.stop; step = pos.step
            if start is None: start = 0
            if stop is None: stop = len(self.x)
            if step is None: step = 1
            return np.stack([self.__call__(self.x[n]) for 
                             n in range(start,stop,step)])
        elif type(pos) == tuple:
            if type(pos[0]) == int:
                return self.__call__(self.x[pos[0]])[pos[1:]]
            elif type(pos[0]) == slice:
                start = pos[0].start; stop = pos[0].stop; step = pos[0].step
                if start is None: start = 0
                if stop is None: stop = len(self.x)
                if step is None: step = 1
                return np.stack([self.__call__(self.x[n])[pos[1:]] for 
                                 n in range(start,stop,step)])
            else:
                raise IndexError('Only integers or slices supported in fCubicSpline.')
        else:
            raise IndexError('Only integers or slices supported in fCubicSpline.')



    def __setitem__(self, pos, val):
        raise RuntimeError("Cannot assign to spline object. "+
                           "Use regenerate_data to recreate original data first.")



    def max(self):
        return self.data_max

    def min(self):
        return self.data_min

    def absmax(self):
        return np.abs(np.array([self.data_max, self.data_min])).max()

    def regenerate_data(self):
        '''
        Rebuild the original data.
        '''
        return np.stack([self.__call__(val) for val in self.x])

