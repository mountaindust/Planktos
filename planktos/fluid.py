'''Functions and methods for loading and handling fluid data.
These are mainly utilized by the Enviornment class.

Created: Thurs July 9 2025

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import warnings
import numpy as np
from scipy import interpolate
from scipy.linalg import solve_banded
from pathlib import Path
from . import _dataio


def _read_IBAMR3d_vtkfiles(path, d_start=0, d_finish=None, 
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
        flow, mesh, time = _dataio.read_vtk_Rectilinear_Grid_Vector(path)
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
            data, mesh, time = _dataio.read_vtk_Rectilinear_Grid_Vector(str(this_file))
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



def _wrap_flow(flow, flow_points, periodic_dim=(True, True, False)):
    '''In some cases, software may print out fluid velocity data that omits 
    the velocities at the right boundaries in spatial dimensions that are 
    meant to be periodic. This helper function restores that data by copying 
    everything over. 3rd dimension will automatically be ignored if 2D.

    This assumes a regular fluid grid.

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

    def __init__(self, flow_times, flow, dydx0=None, dydx1=None, 
                 extrapolate=(True, True), bc_type='not-a-knot', direction=None):
        '''
        Creates a PPoly instance spline instance with some additional info 
        and capabilities. Will throw a custom error if times are requested 
        outside of spline time bounds and extrapolate is False on that side.

        If dydx0 is None then use CubicSpline to construct the object. If 
        bc_type is given as 'left', construct a CubicSpline using not-a-knot 
        and natural BC on the starting side. This is ideal dynamic loading of 
        data when only the first few time points of the data set are currently 
        being splined. If bc_type is something else, it will be passed to 
        scipy.interpolate.CubicSpline.
        
        If dydx0 isn't None, then dydx0 and dydx1 specify derivatives that 
        will be used to extend an old spline either to the left or the right 
        according to the direction argument; see _extend_prev_spline for 
        further info. bc_type will be ignored.
        '''
        if dydx0 is None:
            if bc_type == 'left':
                dydx = self._left_based_cspline(flow_times, flow)
                interpolate.CubicHermiteSpline.__init__(self, flow_times, flow, dydx, 
                                                        axis=0, extrapolate=True)
            else:
                super(fCubicSpline, self).__init__(flow_times, flow, axis=0, 
                                                   extrapolate=True, bc_type=bc_type)
        else:
            assert dydx1 is not None, "dydx1 must be specified with dydx0"
            assert direction is not None, "extension direction must be specified with dydx0"
            dydx = self._extend_prev_spline(flow_times, flow, dydx0, dydx1, direction)
            interpolate.CubicHermiteSpline.__init__(self, flow_times, flow, dydx, 
                                                    axis=0, extrapolate=True)

        self.shape = flow.shape
        self.extrapolate = extrapolate
        # These are inaccurate and should only be used for plotting!
        self.data_max = flow.max()
        self.data_min = flow.min()

    def _extend_prev_spline(self, x, y, dydx0, dydx1, direction='right'):
        '''Set new spline based on derivative data from an old spline.

        Parameters
        ----------
        x : ndarray
            time points corresponding to the flow data
        y : ndarray
            flow data points
        dydx0 : ndarray
            derivatives at first time point
        dydx1 : ndarray
            derivatievs at second time point
        dir : 'right' or 'left'
            if 'right', dydx0 and dydx1 are construed to be at the first and second
            time points respectively (e.g., we are extending a spline to the right).
            Otherwise, they are construed to be the next-to-last and last times 
            (e.g., we are extending a spline to the left).

        Returns
        -------
        ndarray of derivatives to be passed to CubicHermiteSpline

        Notes
        -----
        This implemenation is largely based on the source code scipy.interploate._cubic.py
        '''
        n = len(x)
        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("flow times must be a strictly increasing sequence.")
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr

        # Find derivative values at each x[i] by solving a tridiagonal system.
        A = np.zeros((3, n))  # This is a banded matrix representation.
        b = np.empty((n,) + y.shape[1:], dtype=y.dtype)
        if direction == 'right':
            # Filling the system for i=2..n-1
            #                         (x[i] - x[i-1]) * s[i-2] +\
            # 2 * ((x[i-1] - x[i-2]) + (x[i] - x[i-1])) * s[i-1]   +\
            #                         (x[i-1] - x[i-2]) * s[i] =\
            #       3 * ((x[i] - x[i-1])*(y[i-1] - y[i-2])/(x[i-1] - x[i-2]) +\
            #           (x[i-1] - x[i-2])*(y[i] - y[i-1])/(x[i] - x[i-1]))

            A[-1, :-2] = dx[1:]                  # The lower lower diagonal
            A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The lower diagonal
            A[0, 2:] = dx[:-1]                   # The diagonal

            b[2:] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

            A[0,0] = 1; A[0,1] = 1
            b[0] = dydx0; b[1] = dydx1
            A[1,0] = 0 # derivative of second point is specified.
            l_and_u = (2,0)
        elif direction == 'left':
            # Filling the system for i=0..n-3
            #                         (x[i+2] - x[i+1]) * s[i] +\
            # 2 * ((x[i+1] - x[i]) + (x[i+2] - x[i+1])) * s[i+1]   +\
            #                         (x[i+1] - x[i]) * s[i+2] =\
            #       3 * ((x[i+2] - x[i+1])*(y[i+1] - y[i])/(x[i+1] - x[i]) +\
            #           (x[i+1] - x[i])*(y[i+2] - y[i+1])/(x[i+2] - x[i+1]))

            A[-1, :-2] = dx[1:]                  # The diagonal
            A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The upper diagonal
            A[0, 2:] = dx[:-1]                   # The upper upper diagonal
            
            b[0:-3] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

            A[-1,-2] = 1; A[-1,-1] = 1
            b[-2] = dydx0; b[-1] = dydx1
            A[1,-1] = 0 # derivative of next-to-last point is specified
            l_and_u = (0,2)
        
        # Solve the system
        m = b.shape[0]
        # s is the derivatives of the spline at all data points
        s = solve_banded(l_and_u, A, b.reshape(m,-1), overwrite_ab=True, 
                            overwrite_b=True, check_finite=False)
        s = s.reshape(b.shape)

        return s
    
    def _left_based_cspline(self, x, y):
        '''Create a cubic spline where both boundary conditions are specified
        at the left (natural and 'not-a-knot'). This is extremely useful when 
        only the first part of a fluid data set will be loaded.
        
        Parameters
        ----------
        x : ndarray
            time points corresponding to the flow data
        y : ndarray
            flow data points

        Returns
        -------
        ndarray of derivatives to be passed to CubicHermiteSpline

        Notes
        -----
        This implemenation is largely based on the source code scipy.interploate._cubic.py
        '''
        n = len(x)
        assert n>3, "At least 3 data points are needed for left-based spline."
        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("flow times must be a strictly increasing sequence.")
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr

        # Find derivative values at each x[i] by solving a tridiagonal system.
        A = np.zeros((3, n))  # This is a banded matrix representation.
        b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

        # Filling the system for i=2..n-1
        #                         (x[i] - x[i-1]) * s[i-2] +\
        # 2 * ((x[i-1] - x[i-2]) + (x[i] - x[i-1])) * s[i-1]   +\
        #                         (x[i-1] - x[i-2]) * s[i] =\
        #       3 * ((x[i] - x[i-1])*(y[i-1] - y[i-2])/(x[i-1] - x[i-2]) +\
        #           (x[i-1] - x[i-2])*(y[i] - y[i-1])/(x[i] - x[i-1]))

        A[-1, :-2] = dx[1:]                  # The lower lower diagonal
        A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The lower diagonal
        A[0, 2:] = dx[:-1]                   # The diagonal

        b[2:] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

        d = x[2] - x[0]
        slp = (y[2]-y[0])/d
        # 'not-a-knot' at the start
        A[0, 1] = d
        A[1, 0] = dx[1]
        b[1] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
                dxr[0]**2 * slope[1]) / d
        # natural bc at the start
        A[0, 0] = dx[0]**2 - d**2
        b[0] = slp*dx[0]**2 - slope[0]*d**2
        l_and_u = (2,0)

        # Solve the system
        m = b.shape[0]
        # s is the derivatives of the spline at all data points
        s = solve_banded(l_and_u, A, b.reshape(m,-1), overwrite_ab=True, 
                            overwrite_b=True, check_finite=False)
        s = s.reshape(b.shape)

        return s
        
        

    def __call__(self, val):
        if (val < self.x[0] and not self.extrapolate[0]) \
              or (val > self.x[-1] and not self.extrapolate[1]):
            raise SplineRangeError('Out of range without extrapolation.')
        return super().__call__(val)

    def __getitem__(self, pos):
        '''
        Allows indexing into the interpolator at original time mesh points.
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
    
    def trim_end(self, last_x_idx):
        '''This is used to remove the end of the spline.
        x points up to last_x_idx will be retained.
        '''
        self.c = self.c[:, 0:last_x_idx, ...]
        self.x = self.x[0:last_x_idx+1]
        self.shape = (len(self.x), *self.shape[1:])

    def max(self):
        '''This will return a data max based on the data used to build the spline.'''
        return self.data_max

    def min(self):
        '''This will return a data min based on the data used to build the spline.'''
        return self.data_min

    def absmax(self):
        return np.abs(np.array([self.data_max, self.data_min])).max()

    def regenerate_data(self):
        '''
        Rebuild the original data.
        '''
        return np.stack([self.__call__(val) for val in self.x])
    


class SplineRangeError(ValueError):
    """
    Exception raised for asking for a value outside of interpolation range.
    """
    def __init__(self, message="Value is outside of valid interpolation range."):
        self.message = message
        super().__init__(self.message)
    


class FluidData:
    '''
    Container class for fluid velocity data and its temporal interpolations.

    Routines for manipulating the fluid velocity field should find their way 
    into here from the Environment class.

    The fluid velocity field can be accessed by calling an instance of this 
    class as a function with the desired simulation time as the sole argument. 
    The result will be a list of ndarrays with the following format:
    [x-vel field ndarray (i,j,[k]), y-vel field ndarray (i,j,[k]),
    z-vel field ndarray (if 3D)]. i is x index, j is y index, etc., with the 
    value of x,y, and z increasing as the corresponding index increases.

    If the fluid is not being dynamically loaded from disk, an instance of 
    FluidData will also act as a container object for the raw data arrays. 
    Just access them as though FluidData were a list. If the fluid velocity 
    field is not time-varying, the result will be an ndarray in each direction. 
    If the fluid is time varying, the data is stored as part of an a 
    fCubicSpline object in each direction that can still behave as if it were
    [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k]),
    z-vel field ndarray (if 3D)].

    FluidData is subclassed for loading data from particular types of sources.

    Attributes
    ----------
    flow_points : tuple (len==dimension) of 1D ndarrays
        points defining the spatial grid for the fluid velocity data. These do 
            not have to be evenly spaced, but should have the same length as each 
            spatial dimension of the flow data. It is assumed that endpoints lie 
            on the domain boundary.
    flow_times : ndarray of floats or None
        if specified, the time stamp for each index t in the flow arrays (time 
        varying fluid velocity fields only)
    fshape : shape of each component of the fluid velocity field ([t],i,j,[k])
        as an ndarray of raw data.
    INUM : Number of intervals loaded at any given time when dynamically 
        loading data.
    '''

    def __init__(self, flow, flow_points, flow_times=None, INUM=None):
        '''
        Class file for dynamically loading time-varying fluid data and splining it.

        This object must be called with a time (float). It will then provide a 
        list of fluid ndarrays corresponding to the fluid velocity field at grid 
        points at that time. This interface is purposefully different from the 
        others so that the FluidData object can catch times that are outside of 
        the currently loaded times and dynamically load/spline the data needed. 
        It will hopefully also raise errors where only the old format is 
        supported to aid in debugging.

        Parameters
        ----------
        flow : list of ndarrays
            [x-vel field ndarray ([t],i,j,[k]), y-vel field ndarray ([t],i,j,[k]),
            z-vel field ndarray (if 3D)]. i is x index, j is y index, with the 
            value of x and y increasing as the index increases.
        flow_points : tuple (len==dimension) of 1D ndarrays
            points defining the spatial grid for the fluid velocity data. These do 
            not have to be evenly spaced, but should have the same length as each 
            spatial dimension of the flow data. It is assumed that endpoints lie 
            on the domain boundary.
        flow_times : ndarray of floats
            if specified, the time stamp for each index t in the flow arrays (time 
            varying fluid velocity fields only)
        INUM : int, optional
            Used by subclasses to dynamically load data from storage.
        '''
        if INUM is not None:
            assert INUM % 2 != 0, "INUM must be odd."
        self.INUM = INUM # This is how many intervals to use when initiating 
                         #  the spline object.

        if INUM is not None and len(flow_times) <= INUM:
            raise RuntimeError("Not enough data files for dynamic splining.")

        self.flow_points = flow_points
        self.flow_times = flow_times
        
        if self.flow_times is not None:
            # record shape of the fluid data
            self.fshape = (len(self.flow_times), *flow[n].shape[1:])
            
            if self.INUM is not None and self.INUM < len(self.flow_times)-1:
                ### Create initial spline ###
                load_times = self.flow_times[0:self.INUM+1]
                for n, f in enumerate(flow):
                    flow[n] = fCubicSpline(load_times, f, extrapolate=(True, False), 
                                           bc_type='left')
            else:
                ### Spline all data with not-a-knot ###
                if self.INUM >= len(self.flow_times)-1:
                    warnings.warn(f'{self.INUM} spline intervals with {self.flow_times} '+
                                'time points results in all data being splined. '+
                                'INUM will be set to None in FluidData object.')
                    self.INUM = None
                for n, f in enumerate(flow):
                    flow[n] = fCubicSpline(self.flow_times, f, extrapolate=(True, True))
            self._flow = flow
        else:
            # Time-invariant flow. Just save it as-is.
            self.fshape = flow[0].shape
            self._flow = flow


    def __call__(self, time):
        '''Retrieve fluid data at the requested time and update the spline 
        dynamically as needed.
        '''
        # Enforce constant extrapolation
        if time <= self.flow_times[0]:
            start_time = self.flow_times[0]
            try:
                return [fspline(start_time) for fspline in self._flow]
            except SplineRangeError:
                self.update_spline(start_time)
                return [fspline(start_time) for fspline in self._flow]
            except TypeError:
                print('Cannot pass time to time-invarient flow.')
                raise
        elif time >= self.flow_times[-1]:
            end_time = self.flow_times[-1]
            try:
                return [fspline(end_time) for fspline in self._flow]
            except SplineRangeError:
                self.update_spline(end_time)
                return [fspline(end_time) for fspline in self._flow]
            except TypeError:
                print('Cannot pass time to time-invarient flow.')
                raise
        else:
            # interpolate
            try:
                return [fspline(time) for fspline in self._flow]
            except SplineRangeError:
                self.update_spline(time)
                return [fspline(time) for fspline in self._flow]
            except TypeError:
                print('Cannot pass time to time-invarient flow.')
                raise
    
    def __len__(self):
        '''Returns the len of the fluid list.'''
        return len(self._flow)
    
    def __getitem__(self, pos):
        '''
        Allows direct access to the component fCubicSpline objects and therefore 
        indexing into the interpolators at the original time mesh points as if 
        they were ndarrays. However, behavior is only consistent if all the fluid 
        data has been splined (otherwise, the time index will refer to a shifting 
        time point based on what data is currently loaded and splined). So, allow 
        this if all the data is splined and otherwise return an error.
        '''
        if self.INUM is None:
            return self._flow[pos]
        else:
            raise TypeError('A FluidData object with dynamically loaded data '+
                            'must be called as a function with a simulation '+
                            'time passed as an argument in order to return a '+
                            'list of fluid velocity field ndarrays.')
        
    def load_dumpfiles(self, d_start, d_finish):
        '''Subclasses should implement this method to load additional data.'''
        raise NotImplementedError('The subclass for this type of data must '+
                                  'implement its own data loaders.')
    


    def update_spline(self, time):
        '''The workhorse function for dynamically loading data.'''

        # NOTE: fCubicSpline.c has shape (4,num_of_splines,...) where "..." is 
        #   the array dimensions of the velocity grid.
        while time > self._flow[0].x[-1] and not self._flow[0].extrapolate[1]:
            ### spline forward ###
            # get info about what we will be loading
            d_start = self.loaded_dump_bnds[1]+1 # first dump to load
            idx_start = self.loaded_idx_bnds[1]-1 # first index in new spline
            if self.loaded_dump_bnds[1]-1 + self.INUM > self.d_finish:
                # We are at the end of the dataset.
                d_finish = self.d_finish
                idx_finish = len(self.flow_times)
                extrapolate = (False, True)
            else:
                # We are contained in the middle of the dataset.
                d_finish = self.loaded_dump_bnds[1]-1 + self.INUM
                idx_finish = self.loaded_idx_bnds[1]-1 + self.INUM
                extrapolate = (False, False)
            load_times = self.flow_times[idx_start:idx_finish+1]

            dydx0 = []; dydx1 = []; last_flow = []
            for n in range(len(self._flow)):
                # grab flow at final loaded time from current spline
                last_flow[n] = np.array(self._flow[n](load_times[1]))
                # drop unneeded coefficients to save space before replacement
                self._flow[n].c = self._flow[n].c[:,-1,...]
                # Extract derivative info for next spline
                dydx0.append(self._flow[n].c[2,0,...])
                dx = load_times[1] - load_times[0]
                dydx1.append(3*self._flow[n].c[0,1,...]*dx**2
                             + 2*self._flow[n].c[1,1,...]*dx
                             + self._flow[n].c[2,1,...])
                
            # load new data
            flow = self.load_dumpfiles(d_start, d_finish)
            
            # add old spline data
            for n,f in enumerate(flow):
                flow[n] = np.concatenate((self._flow[n].c[3,0,...],
                                          last_flow[n], f))
            # Remove the rest of the spline data
            self._flow = [0 for n in range(len(flow))]

            # Spline it
            for n in range(len(flow)):
                self._flow[n] = fCubicSpline(load_times, flow[n], dydx0[n], 
                                             dydx1[n], extrapolate, direction='right')
                flow[n] = 0
            self.loaded_dump_bnds = (self.loaded_dump_bnds[1]-1,d_finish)
            self.loaded_idx_bnds = (idx_start, idx_finish)
            
        while time < self._flow[0].x[0] and not self._flow[0].extrapolate[0]:
            ### spline backward ###
            # get info about what we will be loading
            d_finish = self.loaded_dump_bnds[0]-1 # last dump to load
            idx_finish = self.loaded_idx_bnds[0]+1 # last index in new spline
            if self.loaded_dump_bnds[0]+1 - self.INUM < self.d_start:
                # We are at the beginning of the dataset.
                d_start = self.d_start
                idx_start = 0
                extrapolate = (True, False)
            else:
                # We are contained in the middle of the dataset.
                d_start = self.loaded_dump_bnds[0]+1 - self.INUM
                idx_start = self.loaded_idx_bnds[0]+1 - self.INUM
                extrapolate = (False, False)
            load_times = self.flow_times[idx_start:idx_finish+1]

            dydx0 = []; dydx1 = []
            for n in range(len(self._flow)):
                # grab flow at second loaded time from current spline.
                #   this will become the final loaded flow.
                last_flow[n] = np.array(self._flow[n](load_times[-1]))
                # drop unneeded coefficients to save space before replacement
                self._flow[n].c = self._flow[n].c[:,0,...]
                # Extract derivative info for next spline
                dydx0.append(self._flow[n].c[2,0,...])
                dx = load_times[1] - load_times[0]
                dydx1.append(3*self._flow[n].c[0,1,...]*dx**2
                             + 2*self._flow[n].c[1,1,...]*dx
                             + self._flow[n].c[2,1,...])
                
            # load new data
            flow = self.load_dumpfiles(d_start, d_finish)
            
            # add old spline data
            for n,f in enumerate(flow):
                flow[n] = np.concatenate((f, self._flow[n].c[3,0,...],
                                          last_flow[n]))
            # Remove the rest of the spline data
            self._flow = [0 for n in range(len(flow))]

            # Spline it
            for n in range(len(flow)):
                self._flow[n] = fCubicSpline(load_times, flow[n], dydx0[n], 
                                             dydx1[n], extrapolate, direction='left')
                flow[n] = 0
            self.loaded_dump_bnds = (d_start, self.loaded_dump_bnds[0]+1)
            self.loaded_idx_bnds = (idx_start, idx_finish)
    


class IB2dData(FluidData):

    def __init__(self, path, dt, print_dump, d_start=0, d_finish=None, INUM=7):
        '''Reads in vtk flow velocity data generated by IB2d and creates a 
        FluidData instance out of it. Time will be shifted to start at t=0 
        regardless of d_start. Note that the Eulerian grid for IB2d is always 
        regular (not rectilinear).

        Can read in vector data with filenames u.####.vtk or scalar data
        with filenames uX.####.vtk and uY.####.vtk.

        If INUM (interval number) is set to an odd integer >=5, then the data 
        will be dynamically loaded as needed with INUM intervals between the 
        temporal data sets available at any given time.

        IB2d is an Immersed Boundary (IB) code for solving fully coupled
        fluid-structure interaction models in Python and MATLAB. The code is 
        hosted at https://github.com/nickabattista/IB2d

        Parameters
        ----------
        path : str
            path to folder with vtk data
        dt : float
            dt in input2d
        print_dump : int
            print_dump in input2d
        d_start : int, default=0
            number of first vtk dump to read in
        d_finish : int, optional
            number of last vtk dump to read in, or None to read to end
        INUM : odd int >=5 or None (default)
            max number of splined intervals at any one time. Must be odd and 
            at least 5. If it is given as None then all the time-varying
            fluid data will be splined at once. Note the number of time points 
            needed is 1+INUM.
        '''

        ##### Parse parameters and read in data #####
        self.path = path
        d_start = round(d_start)

        path = Path(path)
        if not path.is_dir(): 
            raise FileNotFoundError("Directory {} not found!".format(str(path)))

        #infer d_finish
        file_names = [x.name for x in path.iterdir() if x.is_file()]
        if 'u.' in [x[:2] for x in file_names]:
            u_nums = sorted([int(f[2:6]) for f in file_names if f[:2] == 'u.'])
            if d_finish is None:
                d_finish = u_nums[-1]
            self.vector_data = True
        else:
            assert 'uX.' in [x[:3] for x in file_names],\
                "Could not find u.####.vtk or uX.####.vtk files in {}.".format(str(path))
            u_nums = sorted([int(f[3:7]) for f in file_names if f[:3] == 'uX.'])
            if d_finish is None:
                d_finish = u_nums[-1]
            self.vector_data = False

        # Save time data
        if d_start != d_finish:
            flow_times = np.arange(d_start,d_finish+1)*print_dump*dt
            # shift time so that flow starts at t=0
            flow_times -= self.flow_times[0]
        else:
            flow_times = None

        # Save dump bounds
        self.d_start = d_start
        self.d_finish = d_finish

        ### Load fluid data ###
        if INUM is None:
            print('Reading vtk fluid data...')
            flow, x, y = self._read_IB2d_dumpfiles(self.path, d_start, d_finish, 
                                                   self.vector_data)
            print('Done!')
            
            # shift domain to quadrant 1
            self._orig_flow_points = (x-x[0], y-y[0])
            self.fluid_domain_LLC = (x[0], y[0])

            ### Convert environment dimensions and add back the periodic gridpoints ###
            # IB2d always has periodic BC and returns a VTK with fluid specified 
            #     at grid points but lacking the grid points at the end of the 
            #     domain (since it's a duplicate). Make the fluid periodic within 
            #     Planktos and to fill out the domain by adding back these last points
            flow, flow_points, self.L = _wrap_flow(flow, self._orig_flow_points, 
                                                   periodic_dim=(True, True))
        else:
            flow, x, y = self._read_IB2d_dumpfiles(self.path, d_start, d_start+self.INUM, 
                                                   self.vector_data)
            # shift domain to quadrant 1
            self._orig_flow_points = (x-x[0], y-y[0])
            self.fluid_domain_LLC = (x[0], y[0])

            ### Convert environment dimensions and add back the periodic gridpoints ###
            flow, flow_points, self.L = _wrap_flow(flow, self._orig_flow_points, 
                                                   periodic_dim=(True, True))
            # record the inclusive bounds of the starting dump numbers to be used
            self.loaded_dump_bnds = (self.d_start, self.d_start+self.INUM)
            # same, but based off of zero to correspond with flow_times indices
            self.loaded_idx_bnds = (0,self.INUM)

        # pass to parent to spline the data.
        super().__init__(flow, flow_points, flow_times, INUM)



    def load_dumpfiles(self, d_start, d_finish):
        '''
        Dynamically load additional IB2d data.
        '''
        flow = self._read_IB2d_dumpfiles(self.path, d_start, d_finish, 
                                         self.vector_data, False)
        flow, flow_points, L = _wrap_flow(flow, self._orig_flow_points, 
                                          periodic_dim=(True, True))
        return flow



    def _read_IB2d_dumpfiles(self, path, d_start, d_finish, vector_data, xy=True):
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
        xy : bool
            whether or not to return the x,y grid points as well

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
                strChoice = 'u'
                if xy:
                    uX,uY,x,y = _dataio.read_2DEulerian_Data_From_vtk(path, numSim,
                                                                      strChoice,xy)
                else:
                    uX,uY = _dataio.read_2DEulerian_Data_From_vtk(path, numSim,
                                                                  strChoice,xy)
                X_vel.append(uX.T) # (y,x) -> (x,y) coordinates
                Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates
            else:
                # read in x-directed Velocity Magnitude #
                strChoice = 'uX'
                if xy:
                    uX,x,y = _dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                                   strChoice,xy)
                else:
                    uX = _dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                               strChoice,xy)
                X_vel.append(uX.T) # (y,x) -> (x,y) coordinates

                # read in y-directed Velocity Magnitude #
                strChoice = 'uY'
                uY = _dataio.read_2DEulerian_Data_From_vtk(path,numSim,
                                                           strChoice)
                Y_vel.append(uY.T) # (y,x) -> (x,y) coordinates

            ##### The following is just for reference! ######
            # read in Vorticity #
            # strChoice = 'Omega'; first = 0
            # Omega = _dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                               strChoice,first)
            # read in Pressure #
            # strChoice = 'P'; first = 0
            # P = _dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                           strChoice,first)
            # read in Velocity Magnitude #
            # strChoice = 'uMag'; first = 0
            # uMag = _dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                              strChoice,first)
            # read in x-directed Forces #
            # strChoice = 'Fx'; first = 0
            # Fx = _dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                            strChoice,first)
            # read in y-directed Forces #
            # strChoice = 'Fy'; first = 0
            # Fy = _dataio.read_2DEulerian_Data_From_vtk(pathViz,numSim,
            #                                            strChoice,first)
            ###################################################

        ### Return data ###
        if d_start != d_finish:
            if xy:
                return [np.transpose(np.dstack(X_vel),(2,0,1)), 
                        np.transpose(np.dstack(Y_vel),(2,0,1))] , x, y
            else:
                return [np.transpose(np.dstack(X_vel),(2,0,1)), 
                        np.transpose(np.dstack(Y_vel),(2,0,1))]
        else:
            if xy:
                return [X_vel[0], Y_vel[0]], x, y
            else:
                return [X_vel[0], Y_vel[0]]