'''Functions and methods for loading and handling fluid data.
These are mainly utilized by the Environment class.

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
#####                BASE-LEVEL FLUID DATA CLASSES                #####
#######################################################################


class LinearSpline:
    '''
    Handles dynamic loading for linear interpolation of one dimension of fluid
    data. Returns plain ndarrays.
    '''

    def __init__(self, flow_times, flow, extrapolate=(True, True)):
        '''
        Creates a linear interpolation instance with some additional info
        and capabilities. Will throw a custom error if times are requested
        outside of spline time bounds and extrapolate is False on that side.
        '''
        self.flow_times = flow_times
        self.flow = flow
        self.extrapolate = extrapolate
        self._shape = flow.shape

    @property
    def shape(self):
        # Custom getter for shape property
        return self._shape

    @shape.setter
    def shape(self, value):
        # Make this property read-only
        raise AttributeError("shape is read-only in LinearSpline")

    @property
    def x(self):
        return self.flow_times

    def __call__(self, val):
        if (val < self.flow_times[0] and not self.extrapolate[0]) \
              or (val > self.flow_times[-1] and not self.extrapolate[1]):
            raise SplineRangeError('Out of range without extrapolation.')
        if val <= self.flow_times[0]:
            return self.flow[0]
        if val >= self.flow_times[-1]:
            return self.flow[-1]
        idx = np.searchsorted(self.flow_times, val) - 1
        t0 = self.flow_times[idx]
        t1 = self.flow_times[idx+1]
        f0 = self.flow[idx]
        f1 = self.flow[idx+1]
        return f0 + (f1 - f0) * (val - t0) / (t1 - t0)


    def __getitem__(self, pos):
        '''
        Allows indexing into the interpolator at original time mesh points.
        '''
        if type(pos) == int:
            farray = self.__call__(self.flow_times[pos])
        elif type(pos) == slice:
            start = pos.start; stop = pos.stop; step = pos.step
            if step is None: step = 1
            if step >= 0:
                if start is None: start = 0
                if stop is None: stop = len(self.flow_times)
            else:
                if start is None: start = len(self.flow_times)-1
                if stop is None: stop = -1
            farray = np.stack([self.__call__(self.flow_times[n]) for 
                               n in range(start,stop,step)])
        elif type(pos) == tuple:
            if type(pos[0]) == int:
                farray = self.__call__(self.flow_times[pos[0]])[pos[1:]]
            elif type(pos[0]) == slice:
                start = pos[0].start; stop = pos[0].stop; step = pos[0].step
                if step is None: step = 1
                if step >= 0:
                    if start is None: start = 0
                    if stop is None: stop = len(self.flow_times)
                else:
                    if start is None: start = len(self.flow_times)-1
                    if stop is None: stop = -1
                farray = np.stack([self.__call__(self.flow_times[n])[pos[1:]] for
                                   n in range(start,stop,step)])
            else:
                raise IndexError('Only integers or slices supported in LinearSpline.')
        else:
            raise IndexError('Only integers or slices supported in LinearSpline.')
        
        return farray

    def __setitem__(self, pos, val):
        self.flow[pos] = val

    def max(self):
        return self.flow.max()
    
    def min(self):
        return self.flow.min()
    
    def absmax(self):
        return np.abs(self.flow).max()

    def regenerate_data(self):
        '''
        Return the original data the interpolation is based on.

        Mirrors fCubicSpline.regenerate_data so that both spline classes present
        the same surface. A linear spline stores its raw data outright, so this
        is a direct reference rather than a reconstruction -- deliberately no
        copy, since under dynamic loading the window can be very large.
        '''
        return self.flow

    def derivative(self, val):
        '''
        Returns time derivative at the specified time.
        Note: this is a simple finite difference derivative and piecewise
        constant on right closed intervals (t[i-1], t[i]] -- a time falling
        exactly on a data timestamp takes the slope of the interval to its
        left. The first timestamp is the exception, taking the slope to its
        right, since it has no interval on the left.
        '''
        if (val < self.flow_times[0] and not self.extrapolate[0]) \
              or (val >= self.flow_times[-1] and not self.extrapolate[1]):
            raise SplineRangeError('Out of range without extrapolation.')
        elif val == self.flow_times[0]:
            idx = 0
        elif val < self.flow_times[0] or val >= self.flow_times[-1]:
            return np.zeros_like(self.flow[0]) # constant exrapolation
        else:
            idx = np.searchsorted(self.flow_times, val) - 1
        return (self.flow[idx+1] - self.flow[idx]) / (self.flow_times[idx+1] - self.flow_times[idx])



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
        and natural BC on the starting side. This is for dynamic loading of 
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

        self._shape = flow.shape
        self.extrapolate = extrapolate
        # These are inaccurate and should only be used for plotting!
        self.data_max = flow.max()
        self.data_min = flow.min()

    @property
    def shape(self):
        # Custom getter for shape property
        return self._shape

    @shape.setter
    def shape(self, value):
        # Make this property read-only
        raise AttributeError("shape is read-only in fCubicSpline")



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
        This implementation is largely based on the source code scipy.interpolate._cubic.py
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
        '''
        THIS APPEARS TO NOT WORK. REPLACE WITH LINEAR INTERPOLATION.
        
        Create a cubic spline where both boundary conditions are specified
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
        This implementation is largely based on the source code scipy.interpolate._cubic.py
        '''
        n = len(x)
        assert n>3, "At least 3 data points are needed for left-based spline."
        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("flow times must be a strictly increasing sequence.")
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr

        ##### Old implementation #####
        # # Find derivative values at each x[i] by solving a tridiagonal system.
        # A = np.zeros((3, n))  # This is a banded matrix representation.
        # b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

        # # Filling the system for i=2..n-1
        # #                         (x[i] - x[i-1]) * s[i-2] +\
        # # 2 * ((x[i-1] - x[i-2]) + (x[i] - x[i-1])) * s[i-1]   +\
        # #                         (x[i-1] - x[i-2]) * s[i] =\
        # #       3 * ((x[i] - x[i-1])*(y[i-1] - y[i-2])/(x[i-1] - x[i-2]) +\
        # #           (x[i-1] - x[i-2])*(y[i] - y[i-1])/(x[i] - x[i-1]))

        # A[-1, :-2] = dx[1:]                  # The lower lower diagonal
        # A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The lower diagonal
        # A[0, 2:] = dx[:-1]                   # The diagonal

        # b[2:] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

        # d = x[2] - x[0]
        # slp = (y[2]-y[0])/d
        # # 'not-a-knot' at the start
        # A[0, 1] = d
        # A[1, 0] = dx[1]
        # b[1] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
        #         dxr[0]**2 * slope[1]) / d
        # # natural bc at the start
        # A[0, 0] = dx[0]**2 - d**2
        # b[0] = slp*dx[0]**2 - slope[0]*d**2
        # l_and_u = (2,0)
        ##################################

        A = np.zeros((4, n))  # This is a banded matrix representation.
        b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

        # Filling the system for i=2..n-1
        #                         (x[i] - x[i-1]) * s[i-2] +\
        # 2 * ((x[i-1] - x[i-2]) + (x[i] - x[i-1])) * s[i-1]   +\
        #                         (x[i-1] - x[i-2]) * s[i] =\
        #       3 * ((x[i] - x[i-1])*(y[i-1] - y[i-2])/(x[i-1] - x[i-2]) +\
        #           (x[i-1] - x[i-2])*(y[i] - y[i-1])/(x[i] - x[i-1]))

        A[-1, :-2] = dx[1:]                  # The lower lower diagonal
        A[2, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The lower diagonal
        A[1, 2:] = dx[:-1]                   # The diagonal

        b[2:] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

        # 'not-a-knot' and natural bc at the start
        A[1, 0] = 2
        A[0, 1] = 1 # only thing in the upper diagonal
        b[0] = 3*slope[0]
        A[2, 0] = -(3*dx[1]*dx[0]+dx[1]**2)
        A[1, 1] = 3*dx[0]**2 + 3*dx[0]*dx[1] + dx[1]**2
        b[1] = 3*slope[1]*dx[0]**2
        l_and_u = (2,1)

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
            farray = self.__call__(self.x[pos])
        elif type(pos) == slice:
            start = pos.start; stop = pos.stop; step = pos.step
            if step is None: step = 1
            if step >= 0:
                if start is None: start = 0
                if stop is None: stop = len(self.x)
            else:
                if start is None: start = len(self.x)-1
                if stop is None: stop = -1
            farray = np.stack([self.__call__(self.x[n]) for 
                               n in range(start,stop,step)])
        elif type(pos) == tuple:
            if type(pos[0]) == int:
                farray = self.__call__(self.x[pos[0]])[pos[1:]]
            elif type(pos[0]) == slice:
                start = pos[0].start; stop = pos[0].stop; step = pos[0].step
                if step is None: step = 1
                if step >= 0:
                    if start is None: start = 0
                    if stop is None: stop = len(self.x)
                else:
                    if start is None: start = len(self.x)-1
                    if stop is None: stop = -1
                farray = np.stack([self.__call__(self.x[n])[pos[1:]] for
                                   n in range(start,stop,step)])
            else:
                raise IndexError('Only integers or slices supported in fCubicSpline.')
        else:
            raise IndexError('Only integers or slices supported in fCubicSpline.')
        
        return farray

    def __setitem__(self, pos, val):
        raise RuntimeError("Cannot assign to spline object. "+
                           "Use regenerate_data to recreate original data first.")
    
    def trim_end(self, last_x_idx):
        '''This is used to remove the end of the spline.
        x points up to last_x_idx will be retained.
        '''
        self.c = self.c[:, 0:last_x_idx, ...]
        self.x = self.x[0:last_x_idx+1]
        self._shape = (len(self.x), *self._shape[1:])

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
    

#######################################################################
#####    CONTAINER CLASSES FOR LOADING AND HANDLING FLUID DATA    #####
#######################################################################


# Design note (not user-facing): routines for manipulating the fluid velocity
# field should find their way into this class from the Environment class.
class FluidData:
    '''
    Container class for fluid velocity data and its temporal interpolations.

    ``Environment.flow`` is an instance of this class, or of one of its
    per-source subclasses. It owns the velocity field, the spatial grid the
    field is defined on, the time stamps, and the interpolation in time --
    including loading further data from storage when a requested time falls
    outside what is currently held in memory.

    There are two ways to get velocity data out of it.

    **Call it with a time.** ``envir.flow(t)`` returns the field at simulation
    time ``t``, interpolated in time, as a list of plain ndarrays:
    ``[x-velocity (i,j,[k]), y-velocity (i,j,[k]), z-velocity (3D only)]``,
    where i indexes x, j indexes y and k indexes z, each increasing with the
    corresponding coordinate. This always works, and it is the only option when
    data is being loaded dynamically, since the time requested is what
    determines which data gets loaded.

    **Index it like a list.** ``envir.flow[0]`` gives the x-velocity component.
    For a time-invariant field that is the raw ndarray. For a time-varying field
    it is that component's interpolant in time, which may itself be indexed by
    time index as though it were an ``([t],i,j,[k])`` array. This is unavailable
    while data is being loaded dynamically (``INUM`` an int), because a time
    index would then refer to a position in a shifting window rather than in the
    dataset; ``TypeError`` is raised instead.

    Interpolation in *space* does not happen here. Use
    ``Environment.interpolate_flow``, or one of the ``Swarm`` accessors that
    wrap it (``get_fluid_drift`` and friends).

    Interpolation in *time* is cubic when the whole dataset is held in memory
    and linear when data is loaded dynamically -- see ``INUM`` below, and the
    discussion of the tradeoff in the narrative documentation for this class.

    FluidData is subclassed for loading data from particular types of sources.

    Attributes
    ----------
    flow_points : tuple (len == spatial dimension) of 1D ndarrays
        Points defining the spatial grid the fluid velocity is specified on.
        These need not be evenly spaced, but must have the same length as the
        corresponding spatial dimension of the flow data. Endpoints are assumed
        to lie on the domain boundary.
    flow_times : ndarray of floats, or None
        Time stamp for each index t in the flow arrays. None for a
        time-invariant field.
    fshape : tuple
        Shape of each component of the fluid velocity field as an ndarray of
        raw data, ``([t],i,j,[k])``.
    ndim : int
        Number of spatial dimensions of the fluid velocity field (2 or 3).
    INUM : None, True, or int
        How much data is held in memory, and how it is interpolated in time.
        None (the default) splines the entire dataset cubically. True splines
        the entire dataset linearly. An int loads a sliding window of
        ``INUM`` + 1 time points from storage as needed and splines it linearly.
    periodic_dim : tuple of bool
        Whether the fluid data is periodic in each spatial dimension. Defaults
        to non-periodic, and is independent of the agent boundary conditions in
        ``Environment.bndry``.
    fluid_domain_LLC : tuple
        If the fluid velocity came from data that was translated in space so
        that its lower left corner sat at the origin, this holds the original
        lower left corner.
    fmin : tuple
        Minimum velocity in each direction over all data seen so far.
    fmax : tuple
        Maximum velocity in each direction over all data seen so far.
    '''

    def __init__(self, flow, flow_points, flow_times=None, INUM=None,
                 periodic_dim=False, fluid_domain_LLC=None):
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
        flow_points : tuple (len == spatial dimension) of 1D ndarrays
            points defining the spatial grid for the fluid velocity data. These do 
            not have to be evenly spaced, but should have the same length as each 
            spatial dimension of the flow data. It is assumed that endpoints lie 
            on the domain boundary.
        flow_times : ndarray of floats
            if specified, the time stamp for each index t in the flow arrays (time 
            varying fluid velocity fields only)
        INUM : int, optional
            Used by subclasses to dynamically load data from storage. It corresponds
            to the number of intervals loaded at any given time when dynamically 
            loading data and linearly splining. True results in linearly splining 
            all data, None results in cubic splining all data.
        periodic_dim : bool (default=False), or tuple of bool
            Whether or not the fluid data is periodic in each spatial dimension.
            Periodicity of the fluid data is independent of the agent boundary
            conditions (Environment.bndry); set it to match how the data was
            generated. When True in a dimension, interpolation wraps the upper
            grid edge to the lower edge (so the upper edge must duplicate the
            lower edge, as for genuinely periodic data).
        fluid_domain_LLC : tuple, optional
            If the fluid velocity came from data and was translated in space so 
            that the LLC was in the lower left corner, this stores the original LLC.
        '''
        
        self.INUM = INUM # This is how many intervals to use when initiating 
                         #  the spline object.
        self.fluid_domain_LLC = fluid_domain_LLC

        if INUM is not None and len(flow_times) <= INUM:
            raise RuntimeError("Not enough data files for dynamic splining.")

        # A subclass that streams from storage must hand over the timestamps for
        # the WHOLE dataset, not just the resident window: windows are sliced out
        # of flow_times and the simulation is bounded by its endpoints. Getting
        # this wrong does not raise on its own -- a short flow_times makes
        # INUM >= len(flow_times)-1 below, which quietly selects the "everything
        # is in memory" branch, sets extrapolate=(True, True), and thereby
        # disables update_spline for the rest of the run. Check it here, where
        # the subclass has already published the dump range it intends to cover.
        if INUM is not None and INUM is not True and INUM is not False \
                and flow_times is not None \
                and hasattr(self, 'd_start') and hasattr(self, 'd_finish'):
            expected = self.d_finish - self.d_start + 1
            if len(flow_times) != expected:
                raise RuntimeError(
                    "flow_times must cover the entire dump range when loading "
                    "dynamically: dumps {}-{} is {} time points, but "
                    "flow_times has {}. A loader must timestamp every dump up "
                    "front, not just the window it loads first.".format(
                        self.d_start, self.d_finish, expected, len(flow_times)))

        self.flow_points = flow_points
        self.flow_times = flow_times
        if isinstance(periodic_dim, tuple):
            self.periodic_dim = periodic_dim
        else:
            self.periodic_dim = (periodic_dim,)*len(flow)
        
        if self.flow_times is not None:
            # record shape of the fluid data
            self.fshape = (len(self.flow_times), *flow[0].shape[1:])

            # Per-dump spatial means of each velocity component: the sidecar the
            # plotting statistics read instead of touching the field itself.
            # Recorded here and in update_spline -- wherever data lands in
            # memory -- so it costs one reduction over data that is already
            # resident. NaN marks a dump that has never been loaded, which is
            # possible only when a window is being slid.
            self._dump_means = np.full((len(self.flow_times), len(flow)), np.nan)
            self._record_dump_means(0, flow)
            # Set below for cubic splining, where the whole dataset is resident.
            self._mean_interp = None

            if self.INUM is not None and self.INUM is not False:
                if self.INUM is True or self.INUM >= len(self.flow_times)-1:
                    if self.INUM is not True:
                        # An int INUM is a request for a sliding window, but one
                        # at least as wide as the dataset leaves nothing to
                        # slide. Say so: the resulting object holds everything in
                        # memory and never calls update_spline, which is the
                        # opposite of what was asked for.
                        warnings.warn(
                            "INUM={} spans the entire dataset ({} time points), "
                            "so all fluid data is being held in memory and no "
                            "dynamic loading will occur. Use a smaller INUM to "
                            "load windows from storage.".format(
                                self.INUM, len(self.flow_times)), UserWarning)
                    for n, f in enumerate(flow):
                        flow[n] = LinearSpline(self.flow_times, f, extrapolate=(True, True))
                elif self.INUM < len(self.flow_times)-1:
                    ### Create initial spline ###
                    load_times = self.flow_times[0:self.INUM+1]
                    for n, f in enumerate(flow):
                        flow[n] = LinearSpline(load_times, f, extrapolate=(True, False))
            else:
                ### Spline all data with not-a-knot ###
                self.INUM = None
                for n, f in enumerate(flow):
                    flow[n] = fCubicSpline(self.flow_times, f, extrapolate=(True, True))
                # Spline the per-dump means with the same class and knots as the
                # field. Cubic spline construction is linear in the data, and so
                # is the spatial mean, so this evaluates to exactly the mean of
                # the splined field at any time -- provided the two are built
                # the same way, which reusing fCubicSpline guarantees.
                self._mean_interp = fCubicSpline(self.flow_times, self._dump_means,
                                                 extrapolate=(True, True))
            self._flow = flow
        else:
            # Time-invariant flow. Just save it as-is.
            self.fshape = flow[0].shape
            self._flow = list(flow)
            self._dump_means = np.array([np.mean(f) for f in self._flow])
            self._mean_interp = None

        self.fmin = tuple(f.min() for f in self._flow)
        self.fmax = tuple(f.max() for f in self._flow)


    def __call__(self, time):
        '''Retrieve fluid data at the requested time and update the spline 
        dynamically as needed.
        '''
        # Enforce constant extrapolation beyond full time bounds
        if time <= self.flow_times[0]:
            start_time = self.flow_times[0]
            try:
                return [fspline(start_time) for fspline in self._flow]
            except SplineRangeError:
                self.update_spline(start_time)
                return [fspline(start_time) for fspline in self._flow]
            except TypeError:
                print('Cannot pass time to time-invariant flow.')
                raise
        elif time >= self.flow_times[-1]:
            end_time = self.flow_times[-1]
            try:
                return [fspline(end_time) for fspline in self._flow]
            except SplineRangeError:
                self.update_spline(end_time)
                return [fspline(end_time) for fspline in self._flow]
            except TypeError:
                print('Cannot pass time to time-invariant flow.')
                raise
        else:
            # interpolate within full time bounds
            try:
                return [fspline(time) for fspline in self._flow]
            except SplineRangeError:
                self.update_spline(time)
                return [fspline(time) for fspline in self._flow]
            except TypeError:
                print('Cannot pass time to time-invariant flow.')
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

    @property
    def ndim(self):
        '''Returns the number of dimensions of the fluid velocity field.'''
        return len(self.flow_points)
    


    def get_raw_loaded_data(self):
        '''Get the ndarrays that the current splines are based on.

        Under dynamic loading this is only the currently loaded window, not the
        whole dataset.
        '''
        # Static flow is stored as arrays directly; time-varying flow is always
        # held as splines, of either class. Testing flow_times rather than the
        # spline type avoids the old bug where the non-cubic branch assumed
        # "static" and so handed back LinearSpline objects instead of ndarrays
        # on the dynamic-loading path.
        if self.flow_times is None:
            return self._flow
        return [flow.regenerate_data() for flow in self._flow]



    def _record_dump_means(self, idx_start, flow):
        '''Cache the spatial mean of each velocity component for loaded dumps.

        Called wherever fluid data arrives in memory, so the reduction is over
        data that is already resident and costs nothing extra.

        Parameters
        ----------
        idx_start : int
            index into flow_times that the first time point of the passed data
            corresponds to
        flow : list of ndarrays
            per-component fluid data with a leading time axis, in the form
            load_dumpfiles returns
        '''

        for n, f in enumerate(flow):
            means = np.mean(f, axis=tuple(range(1, f.ndim)))
            self._dump_means[idx_start:idx_start+len(means), n] = means



    def _interp_dump_means(self, time):
        '''Evaluate the per-dump mean sidecar at a time, or None if data is missing.

        Returns None when a dump bracketing the requested time has never been
        loaded, so its mean was never recorded -- the caller decides whether to
        pay for a load.

        This mirrors the temporal interpolation of the field itself, and is
        exact rather than approximate: both spline classes evaluate as a
        weighted sum of the nodal fields, and the spatial mean is linear, so
        mean(u(t)) == sum_i w_i(t)*mean(u_i). Times outside the data bounds get
        the same constant extrapolation __call__ applies.
        '''

        if self._mean_interp is not None:
            # Cubic. The entire dataset was resident when the interpolant was
            # built, so no mean can be missing.
            if time <= self.flow_times[0]:
                time = self.flow_times[0]
            elif time >= self.flow_times[-1]:
                time = self.flow_times[-1]
            return tuple(float(m) for m in self._mean_interp(time))

        # Linear. Done off flow_times and the sidecar rather than off the
        # resident spline, so it stays correct for any dump whose mean has been
        # recorded -- including one the sliding window has since moved past.
        if time <= self.flow_times[0]:
            means = self._dump_means[0]
        elif time >= self.flow_times[-1]:
            means = self._dump_means[-1]
        else:
            idx = np.searchsorted(self.flow_times, time) - 1
            m0 = self._dump_means[idx]
            m1 = self._dump_means[idx+1]
            means = m0 + (m1 - m0) * (time - self.flow_times[idx]) / (
                    self.flow_times[idx+1] - self.flow_times[idx])

        if np.isnan(means).any():
            return None
        return tuple(float(m) for m in means)



    def get_mean_velocity(self, time=None, t_idx=None):
        '''Spatial mean of each fluid velocity component.

        This is served from a per-dump cache of means built as data loads, so it
        does not touch the velocity field and does not trigger a load for any
        time whose bracketing dumps have already been seen. That matters for
        plotting, which asks for these once per frame: under dynamic loading,
        computing them from the field would re-stream the entire dataset.

        The value is exact, not approximate -- see the note on linearity in
        docs/notes/flow_field_interface.md §8.5.

        Parameters
        ----------
        time : float, optional
            The time at which to evaluate the means. Ignored, with a warning,
            for time-invariant flow.
        t_idx : int, optional
            The index into flow_times at which to evaluate the means.

        Returns
        -------
        tuple of floats, one per velocity component
        '''

        if self.flow_times is None:
            if time is not None or t_idx is not None:
                warnings.warn("Flow is time-invariant; ignoring time and t_idx.")
            return tuple(float(m) for m in self._dump_means)

        if time is None and t_idx is not None:
            time = self.flow_times[t_idx]
        elif time is None and t_idx is None:
            raise ValueError("Either time or t_idx must be specified.")

        means = self._interp_dump_means(time)
        if means is None:
            # A bracketing dump has never been in memory. Load it -- which
            # records the means for the whole window -- and try again.
            flow = self(time)
            means = self._interp_dump_means(time)
            if means is None:
                # The load did not cover both bracketing dumps. Reduce the field
                # we now hold; it is the same value, just paid for the hard way.
                return tuple(float(np.mean(f)) for f in flow)
        return means



    def load_dumpfiles(self, d_start, d_finish):
        '''Subclasses should implement this method to load additional data.'''
        raise NotImplementedError('The subclass for this type of data must '+
                                  'implement its own data loaders.')



    def update_spline(self, time):
        '''The workhorse function for dynamically loading data. Responds to
        requests for times outside of the currently loaded time interval by 
        loading new data and creating a new spline that includes the new data.
        '''

        while time > self._flow[0].x[-1] and not self._flow[0].extrapolate[1]:
            # spline forward

            ####### get info about what we will be loading #######
            d_start = self.loaded_dump_bnds[1]+1 # first dump to load
            idx_start = self.loaded_idx_bnds[1]-1 # first index in new spline
            if self.loaded_dump_bnds[1]-1 + self.INUM >= self.d_finish:
                # We are at the end of the dataset. Note >=, not >: a window
                # reaching exactly the last dump IS the final one, and used to
                # fall through to the middle branch and be flagged
                # extrapolate=(False, False) -- claiming there was more data to
                # the right when there was none. Latent rather than live, since
                # __call__ clamps to flow_times[-1] and so never asks for a time
                # past a window that already ends there. But it made the flag
                # dishonest for anything else reading it, and left the degenerate
                # "slide past the end" path reachable by a direct update_spline
                # call, where load_dumpfiles would be handed d_start > d_finish.
                # Triggered only when the arithmetic lands exactly on the last
                # index, which is why it survived: forward slides advance the
                # window end by INUM-1, so with INUM=4 it needs a dump count
                # congruent to 2 mod 3 -- 17, 20, 23... The real OpenFOAM series
                # has 17, which is how it was finally found.
                d_finish = self.d_finish
                # Last valid index, matching the inclusive convention used by
                # loaded_dump_bnds and by the middle branch below. (This was
                # len(self.flow_times), which named a nonexistent index. The
                # window itself was unaffected -- the slice below clips -- but
                # loaded_idx_bnds then disagreed with loaded_dump_bnds.)
                idx_finish = len(self.flow_times) - 1
                extrapolate = (False, True)
            else:
                # We are contained in the middle of the dataset.
                d_finish = self.loaded_dump_bnds[1]-1 + self.INUM
                idx_finish = self.loaded_idx_bnds[1]-1 + self.INUM
                extrapolate = (False, False)
            load_times = self.flow_times[idx_start:idx_finish+1]

            ####### retain only the necessary current data #######
            last_flow_0 = []; last_flow_1 = []
            for n in range(len(self._flow)):
                # grab flow at holdover times
                last_flow_0.append(np.array(self._flow[n](load_times[0])))
                # grab flow at final loaded time from current spline
                last_flow_1.append(np.array(self._flow[n](load_times[1])))
                # free up memory
                self._flow[n] = None
                
            # load new data
            flow = self.load_dumpfiles(d_start, d_finish)

            # Record means for the freshly loaded dumps, which start two time
            # points into the new window. The two holdovers prepended below are
            # already in the sidecar from when they were first loaded, and those
            # entries came from raw data rather than from a spline evaluation
            # carried across a window boundary.
            self._record_dump_means(idx_start+2, flow)

            # add old spline data
            for n,f in enumerate(flow):
                flow[n] = np.concatenate((last_flow_0[n][np.newaxis,...],
                                          last_flow_1[n][np.newaxis,...], f))

            ####### Spline it #######
            for n in range(len(flow)):
                self._flow[n] = LinearSpline(load_times, flow[n], extrapolate)
            self.loaded_dump_bnds = (self.loaded_dump_bnds[1]-1,d_finish)
            self.loaded_idx_bnds = (idx_start, idx_finish)

            # Update fmin/fmax
            self.fmin = tuple(min(self.fmin[n],f.min()) for n,f in enumerate(self._flow))
            self.fmax = tuple(max(self.fmax[n],f.max()) for n,f in enumerate(self._flow))
            
        while time < self._flow[0].x[0] and not self._flow[0].extrapolate[0]:
            # spline backward

            ####### if the beginning is requested, jump there #######
            if time <= self.flow_times[self.INUM]:
                self._flow = None
                self._flow = self.load_dumpfiles(self.d_start, self.d_start + self.INUM)
                self.loaded_dump_bnds = (self.d_start, self.d_start + self.INUM)
                self.loaded_idx_bnds = (0, self.INUM)
                self._record_dump_means(0, self._flow)
                for n in range(len(self._flow)):
                    self._flow[n] = LinearSpline(
                        self.flow_times[0:self.INUM+1], self._flow[n],
                        extrapolate=(True, False))
            else:
            ####### We are contained in the middle of the dataset. #######
                ####### get info about what we will be loading #######
                d_finish = self.loaded_dump_bnds[0]-1 # last dump to load
                idx_finish = self.loaded_idx_bnds[0]+1 # last index in new spline
                d_start = self.loaded_dump_bnds[0]+1 - self.INUM
                idx_start = self.loaded_idx_bnds[0]+1 - self.INUM
                extrapolate = (False, False)
                load_times = self.flow_times[idx_start:idx_finish+1]

                ####### retain only the necessary current data #######
                last_flow_0 = []; last_flow_1 = []
                for n in range(len(self._flow)):
                    # grab flow at second loaded time from current spline.
                    #   this will become the final loaded flow.
                    last_flow_1.append(np.array(self._flow[n](load_times[-1])))
                    # grab flow at first loaded time, will be next to last flow.
                    last_flow_0.append(np.array(self._flow[n](load_times[-2])))
                    # free up memory
                    self._flow[n] = None
                    
                # load new data
                flow = self.load_dumpfiles(d_start, d_finish)

                # Record means for the freshly loaded dumps. Sliding backward,
                # these occupy the front of the new window; the two holdovers
                # appended below already have their means recorded.
                self._record_dump_means(idx_start, flow)

                # add old spline data
                for n,f in enumerate(flow):
                    flow[n] = np.concatenate((f, last_flow_0[n][np.newaxis,...],
                                              last_flow_1[n][np.newaxis,...]))

                ####### Spline it #######
                for n in range(len(flow)):
                    self._flow[n] = LinearSpline(load_times, flow[n], extrapolate)
                self.loaded_dump_bnds = (d_start, self.loaded_dump_bnds[0]+1)
                self.loaded_idx_bnds = (idx_start, idx_finish)

            # Update fmin/fmax
            self.fmin = tuple(min(self.fmin[n],f.min()) for n,f in enumerate(self._flow))
            self.fmax = tuple(max(self.fmax[n],f.max()) for n,f in enumerate(self._flow))
    


    def tile_flow(self, x=1, y=1):
        '''Tile the fluid flow a number of times in the x and/or y directions.

        .. note::
           **Temporarily unavailable.** Tiling was previously implemented as a
           virtual view (the ``FlowArray`` ndarray subclass) that reported a
           tiled ``shape`` while storing a single tile. That approach is
           defeated by modern scipy: ``RegularGridInterpolator`` calls
           ``np.asarray`` on any array-API object, which discards the virtual
           shape and hands the interpolator the untiled buffer. ``FlowArray``
           has therefore been removed, and tiling with it.

           It will return as a position-wrapping implementation that works in
           both 2D and 3D without materializing the tiled field, after the
           plotting work. See ``docs/notes/flow_field_interface.md`` for the
           design and the reasoning -- and §9.1 there for the checklist of every
           notice and replaced test to undo when it lands.

           The previous body is preserved commented-out below. Its
           ``flow_points`` extension in particular carries over unchanged: the
           reported coordinate arrays still have to grow with the tiling even
           though the velocity data will not.

        Parameters
        ----------
        x : int, default=1
            number of tiles in the x direction (counting the one already there)
        y : int, default=1
            number of tiles in the y direction (counting the one already there)

        Raises
        ------
        NotImplementedError
            always, until the position-wrapping implementation lands
        '''

        raise NotImplementedError(
            'Tiling is temporarily unavailable. The previous implementation '
            'relied on the FlowArray view, which modern scipy defeats by '
            'coercing array-API objects with np.asarray; FlowArray has been '
            'removed. Tiling will return as a position-wrapping implementation '
            'for 2D and 3D. See docs/notes/flow_field_interface.md.')

        # --- PREVIOUS IMPLEMENTATION, KEPT FOR RESTORATION ------------------
        # Retained deliberately rather than left to git history. The `f.tiling`
        # propagation is dead (FlowArray and the spline `tiling` attributes are
        # gone), but the fshape arithmetic and the flow_points extension are the
        # shape/geometry half of the "public geometry reflects the tiled domain,
        # stored data stays the base tile" rule, and carry over as-is.
        #
        # TIME_DEP = self.flow_times is not None
        #
        # self.tiling = (x,y)
        # new_flow_shape = list(self.fshape)
        #
        # # get new dimensions and pass to flow objects
        # if not TIME_DEP:
        #     for dim,tnum in enumerate(self.tiling):
        #         new_flow_shape[dim] += (self.fshape[dim]-1)*(tnum-1)
        #     self.fshape = tuple(new_flow_shape)
        #     # Update tiling of FlowArray objects
        #     for f in self._flow:
        #         f.tiling = self.tiling
        # else:
        #     for dim,tnum in enumerate(self.tiling):
        #         new_flow_shape[dim+1] += (self.fshape[dim+1]-1)*(tnum-1)
        #     self.fshape = tuple(new_flow_shape)
        #     # Update tiling of fCubicSpline objects
        #     for f in self._flow:
        #         f.tiling = self.tiling
        #         assert f.shape[1:] == self.fshape[1:], "Tiling did not propagate correctly"
        #
        # # extend flow_points
        # flow_points = []
        # for d,fp in enumerate(self.flow_points[:2]):
        #     flow_points.append(np.concatenate(
        #         [fp] + [fp[1:]+fp[-1]*n for n in range(1,self.tiling[d])]
        #         ))
        # if len(self.flow_points) == 3:
        #     flow_points.append(self.flow_points[2])
        # self.flow_points = tuple(flow_points)
        # --------------------------------------------------------------------
    


    def get_vorticity(self, time=None, t_idx=None):
        '''Compute the vorticity field from the fluid velocity field.

        If the flow is time-varying, the vorticity will be computed at 
        the specified time or time index.

        Parameters
        ----------
        time : float, optional
            The time at which to compute the vorticity.
        t_idx : int, optional
            The time index at which to compute the vorticity.
        '''

        if self.flow_times is not None:
            if time is None and t_idx is not None:
                time = self.flow_times[t_idx]
            elif time is None and t_idx is None:
                raise ValueError("Either time or t_idx must be specified.")
            flow = self(time)
        else:
            if time is not None or t_idx is not None:
                warnings.warn("Flow is time-invariant; ignoring time and t_idx.")
            flow = self

        if self.ndim == 2:
            dvydx = np.gradient(flow[1][:], self.flow_points[0], axis=0)
            dvxdy = np.gradient(flow[0][:], self.flow_points[1], axis=1)

            vort = dvydx - dvxdy
        else:
            # Handle 3D case
            dvxdy = np.gradient(flow[0][:], self.flow_points[1], axis=1)
            dvxdz = np.gradient(flow[0][:], self.flow_points[2], axis=2)
            dvydx = np.gradient(flow[1][:], self.flow_points[0], axis=0)
            dvydz = np.gradient(flow[1][:], self.flow_points[2], axis=2)
            dvzdx = np.gradient(flow[2][:], self.flow_points[0], axis=0)
            dvzdy = np.gradient(flow[2][:], self.flow_points[1], axis=1)

            vort = (dvzdy - dvydz, dvxdz - dvzdx, dvydx - dvxdy)

        return vort
    


    def get_dudt(self, time=None, t_idx=None):
        '''Compute the derivative of the fluid velocity with respect to time.

        If the flow is time-varying, the derivative will be computed at the
        specified time or time index, one of which must be provided.

        Parameters
        ----------
        time : float, optional
            The time at which to compute the derivative.
        t_idx : int, optional
            The time index at which to compute the derivative.

        Returns
        -------
        list of ndarrays
            The time derivative of the fluid velocity field.
        '''

        if self.flow_times is not None:
            if time is None and t_idx is not None:
                time = self.flow_times[t_idx]
            elif time is None and t_idx is None:
                raise ValueError("Either time or t_idx must be specified.")
        else:
            # temporally constant flow
            warnings.warn("Flow is time-invariant; returning zero derivative.")
            return [np.zeros(self.fshape) for ii in range(len(self))]
        
        # Constant extrapolation strictly beyond the data's time bounds: the
        # velocity is held constant there, so du/dt = 0. At the endpoints
        # themselves the spline derivative is well-defined, so use strict
        # inequalities. fshape[1:] drops the leading time axis, giving a single-
        # time field per component (fshape includes the time axis for time-
        # varying flow).
        if time < self.flow_times[0] or time > self.flow_times[-1]:
            return [np.zeros(self.fshape[1:]) for ii in range(len(self))]
        else:
            dudt_list = []
            # ensure the relevant data is loaded
            try:
                self._flow[0](time)
            except SplineRangeError:
                self.update_spline(time)
            except TypeError:
                print('Cannot pass time to time-invariant flow.')
                raise
            for fspline in self._flow:
                if isinstance(fspline, fCubicSpline):
                    dudt = fspline.derivative()(time)
                else:
                    # LinearSpline
                    dudt = fspline.derivative(time)
                dudt_list.append(dudt)

        return dudt_list
    


    def calculate_DuDt(self, time=None, t_idx=None):
        '''Compute the material derivative of the fluid velocity field. 
        Gradient is calculated via second order accurate central differences 
        (using numpy) with second order accuracy at the boundaries.

        If the flow is time-varying, the material derivative will be computed 
        at the specified time or time index, one of which must be provided.

        The material derivative is given by
        .. math::
        \\frac{D\\mathbf{u}}{Dt} = \\mathbf{u}_t + 
        (\\nabla\\mathbf{u})\\mathbf{u}

        Parameters
        ----------
        time : float, optional
            The time at which to compute the material derivative.
        t_idx : int, optional
            The time index at which to compute the material derivative.

        Returns
        -------
        list of ndarrays
            The material derivative of the fluid velocity field.
        '''

        if self.flow_times is not None:
            if time is None and t_idx is not None:
                time = self.flow_times[t_idx]
            elif time is None and t_idx is None:
                raise ValueError("Either time or t_idx must be specified.")
            flow = self(time)
        else:
            if time is not None or t_idx is not None:
                warnings.warn("Flow is time-invariant; ignoring time and t_idx.")
            # temporally constant flow
            flow = self

        if self.ndim == 3:
            axis_tuple = (1,2,3)
        else:
            axis_tuple = (1,2)

        flow_grad = np.gradient(np.array(flow), *self.flow_points, edge_order=2, 
                                axis=axis_tuple)

        # Take dot product
        DuDt = []
        for g,f in zip(flow_grad,flow):
            DuDt.append(g*f)
        DuDt = np.sum(DuDt, axis=0)

        # Add dudt
        DuDt += np.array(self.get_dudt(time))

        return [u for u in DuDt]



class IB2dData(FluidData):

    def __init__(self, path, dt, print_dump, d_start=0, d_finish=None, INUM=None):
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
        INUM : int > 3 or None (default)
            max number of linearly splined intervals at any one time. Must be 
            at least 4. If it is given as True, then all time-varying
            fluid data will be linearly splined at once. If None, all will be 
            cubically splined instead. Note the number of data sets 
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
            flow_times -= flow_times[0]
        else:
            flow_times = None

        # Save dump bounds
        self.d_start = d_start
        self.d_finish = d_finish

        ### Load fluid data ###
        if INUM is None or INUM is True:
            print('Reading vtk fluid data...')
            flow, x, y = self._read_IB2d_dumpfiles(self.path, self.d_start, 
                                                   self.d_finish, self.vector_data)
            print('Done!')
            
            # shift domain to quadrant 1
            self._orig_flow_points = (x-x[0], y-y[0])
            fluid_domain_LLC = (x[0], y[0])

            ### Convert environment dimensions and add back the periodic gridpoints ###
            # IB2d always has periodic BC and returns a VTK with fluid specified 
            #     at grid points but lacking the grid points at the end of the 
            #     domain (since it's a duplicate). Make the fluid periodic within 
            #     Planktos and to fill out the domain by adding back these last points
            flow, flow_points, self.L = _wrap_flow(flow, self._orig_flow_points, 
                                                   periodic_dim=(True, True))
        else:
            assert INUM > 3, 'INUM must be at least 4.'
            flow, x, y = self._read_IB2d_dumpfiles(self.path, self.d_start, 
                                                   self.d_start+INUM, 
                                                   self.vector_data)
            # shift domain to quadrant 1
            self._orig_flow_points = (x-x[0], y-y[0])
            fluid_domain_LLC = (x[0], y[0])

            ### Convert environment dimensions and add back the periodic gridpoints ###
            flow, flow_points, self.L = _wrap_flow(flow, self._orig_flow_points, 
                                                   periodic_dim=(True, True))
            # record the inclusive bounds of the starting dump numbers to be used
            self.loaded_dump_bnds = (self.d_start, self.d_start+INUM)
            # same, but based off of zero to correspond with flow_times indices
            self.loaded_idx_bnds = (0,INUM)

        # pass to parent to spline the data.
        super().__init__(flow, flow_points, flow_times, INUM, periodic_dim=True,
                         fluid_domain_LLC=fluid_domain_LLC)



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
            


class VTK3dData(FluidData):

    def __init__(self, path, title='IBAMR_db_', d_start=0, d_finish=None, 
                 INUM=7, periodic_dim=(True, True, False), vel_conv=None):
        '''Reads in one or more vtk Rectilinear Grid Vector files. If path
        refers to a single file, the resulting flow will be time invariant.
        Otherwise, this method will assume that files are named <title>###.vtk 
        where ### is the dump number, and that the mesh is the same in each vtk.
        Also, imported times will be translated backward so that the first time 
        loaded corresponds to a Planktos environment time of 0.0.

        If INUM (interval number) is set to an odd integer >=5, then the data 
        will be dynamically loaded as needed with INUM intervals between the 
        temporal data sets available at any given time.

        It is assumed that the fluid spatial grid includes all domain boundaries.

        Parameters
        ----------
        path : string
            path to vtk data. This can either be a directory or a single file.
            If it is a single file, other parameters except vel_conv are ignored.
        title : string, optional
            The name of each vtk before the dump number. Defaults to
            ``IBAMR_db_``.
        d_start : int, default=0
            vtk dump number to start with.
        d_finish : int, optional
            vtk dump number to end with. If None, end with last one.
        INUM : int > 3 or None (default)
            max number of splined intervals at any one time. Must be  
            at least 4. If it is given as None then all the time-varying
            fluid data will be splined at once. Note the number of time points 
            needed is 1+INUM.
        periodic_dim : list of 2 or 3 bool, default=(True, True, False)
            True if that spatial dimension is periodic, otherwise False
        vel_conv : float, optional
            scalar to multiply the velocity by in order to convert units to 
            match the spatial grid units
        '''

        ##### Parse parameters and read in data #####
        self.path = path
        self.title = title
        self.vel_conv = vel_conv

        path = Path(path)
        if path.is_file():
            flow, mesh, time = _dataio.read_vtk_Rectilinear_Grid_Vector(path)
            flow_times = None
        
        elif path.is_dir():
            tlen = len(title)
            file_names = [x.name for x in path.iterdir() if x.is_file() and
                      x.name[:tlen] == title]
            # get number width
            self.nwidth = len(file_names[0])-len(title)-len('.vtk')
            # get file numbers and store d_start, d_finish.
            file_nums = sorted([int(f[tlen:-4]) for f in file_names])
            if d_start is None:
                self.d_start = file_nums[0]
            else:
                self.d_start = round(d_start)
                assert d_start in file_nums, "d_start number not found!"
            if d_finish is None:
                self.d_finish = file_nums[-1]
            else:
                self.d_finish = round(d_finish)
                assert d_finish in file_nums, "d_finish number not found!"
            
            ### Timestamp the whole dump series before loading any fluid ###
            # flow_times must span the ENTIRE range, not just whatever window is
            # resident: FluidData slices windows out of it and bounds the
            # simulation by its endpoints. Building it from the opening window
            # alone used to make INUM >= len(flow_times)-1, which sends FluidData
            # down its "everything is already in memory" branch with
            # extrapolate=(True, True) -- disabling update_spline outright and
            # silently freezing the fluid at the end of the first window.
            flow_times = self._read_all_times(self.d_start, self.d_finish)

            ### Load fluid data ###
            if INUM is None or INUM is True:
                print('Reading vtk fluid data...')
                flow, mesh = self._read_vtkfiles(self.path, self.title,
                                                 self.d_start, self.d_finish)
                print('Done!')
            else:
                assert INUM > 3, 'INUM must be at least 4.'
                flow, mesh = self._read_vtkfiles(self.path, self.title,
                                                 self.d_start, self.d_start+INUM)
                # record the inclusive bounds of the starting dump numbers to be used
                self.loaded_dump_bnds = (self.d_start, self.d_start+INUM)
                # same, but based off of zero to correspond with flow_times indices
                self.loaded_idx_bnds = (0,INUM)
        else:
            raise FileNotFoundError("Directory {} not found!".format(str(path)))
    
        # shift domain to quadrant 1
        flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                        mesh[2]-mesh[2][0])
        fluid_domain_LLC = (mesh[0][0], mesh[1][0], mesh[2][0])
        # It is assumed that the fluid spatial grid includes all 
        # domain boundaries.
        self.L = [flow_points[0][-1], flow_points[1][-1], flow_points[2][-1]]
        
        if self.vel_conv is not None:
            print("Converting vel units by a factor of {}.".format(self.vel_conv))
            for ii, d in enumerate(flow):
                flow[ii] = d*self.vel_conv

        super().__init__(flow, flow_points, flow_times, INUM, periodic_dim,
                         fluid_domain_LLC=fluid_domain_LLC)
        


    def load_dumpfiles(self, d_start, d_finish):
        '''
        Dynamically load additional data.
        '''
        flow, mesh = self._read_vtkfiles(self.path, self.title,
                                         d_start, d_finish)
        if self.vel_conv is not None:
            for ii, d in enumerate(flow):
                flow[ii] = d*self.vel_conv
        return flow



    def _read_all_times(self, d_start, d_finish):
        '''Timestamp every dump in [d_start, d_finish] without parsing the files.

        TIME sits in each file's header, so this costs one small header read per
        dump rather than a full parse -- see _dataio.read_vtk_time_only. Any file
        whose header scan comes up empty is read in full before concluding it is
        untimed, since the format permits FIELD data outside the header.

        Parameters
        ----------
        d_start : int
            first dump number in the series
        d_finish : int
            last dump number in the series (inclusive)

        Returns
        -------
        ndarray of times shifted so that d_start is at t=0, or None if the
            series holds a single dump (time-invariant flow). If time
            information is missing from any dump, warns and falls back to unit
            time steps across the whole series.
        '''

        if d_start == d_finish:
            return None

        path = Path(self.path)
        times = []
        for n in range(d_start, d_finish+1):
            fname = str(path / (self.title + str(n).zfill(self.nwidth) + '.vtk'))
            time = _dataio.read_vtk_time_only(fname)
            if time is None:
                # Header scan missed it; pay for a full read of this one file
                # before deciding the dump carries no time information.
                time = _dataio.read_vtk_Rectilinear_Grid_Vector(fname)[2]
            times.append(time)

        if None in times:
            warnings.warn("Could not retrieve time information from at least"+
                          " one vtk file. Assuming unit time-steps...", UserWarning)
            return np.arange(len(times), dtype=float)

        times = np.array(times, dtype=float)
        # shift so that the first dump loaded corresponds to environment time 0
        return times - times[0]


    
    def _read_vtkfiles(self, path, title, d_start, d_finish):
        '''Reads in one or more vtk Rectilinear Grid Vector files. If path
        refers to a single file, the resulting flow will be time invariant.
        Otherwise, this method will assume that files are named IBAMR_db_###.vtk 
        where ### is the dump number, and that the mesh is the same in each vtk.
        Also, imported times will be translated backward so that the first time 
        loaded corresponds to a Planktos environment time of 0.0.

        Parameters
        ----------
        path : string
            path to vtk data, incl. file extension if a single file
        title : string
            The name of each vtk before the dump number
        d_start : int, default=0
            vtk dump number to start with.
        d_finish : int, optional
            vtk dump number to end with. If None, end with last one.

        Returns
        -------
        flow : list of ndarray (fluid data)
        mesh : list of 1D arrays of grid points in x, y, and z directions

        Notes
        -----
        Time information is deliberately not returned here. The timeline for the
        whole series is built once, up front, by _read_all_times; this method is
        also the per-window loader on the dynamic path, where re-deriving times
        from the resident window is both wasted work and how flow_times came to
        describe only part of the dataset.
        '''

        path = Path(path)

        ### Gather data ###
        flow = [[], [], []]

        for n in range(d_start, d_finish+1):
            num = str(n).zfill(self.nwidth)
            this_file = path / (title+num+'.vtk')
            data, mesh, _ = _dataio.read_vtk_Rectilinear_Grid_Vector(str(this_file))
            for dim in range(3):
                flow[dim].append(data[dim])

        flow = [np.array(flow[0]).squeeze(), np.array(flow[1]).squeeze(),
                np.array(flow[2]).squeeze()]

        return flow, mesh



class ComsolVTUData(FluidData):

    def __init__(self, path, periodic_dim=(False, False, False), res=101,
                 linear_interp=False, vel_conv=None):
        '''Reads in one or more vtu Rectilinear Grid Vector files exported from 
        COMSOL Multiphysics. It is assumed that the fluid spatial grid includes 
        all domain boundaries.

        Parameters
        ----------
        path : string
            path to folder with vtu data.
        time_points : list or ndarray
            list of times corresponding to each vtu file.
        periodic_dim : list of 2 or 3 bool, default=(False, False, False)
            True if that spatial dimension is periodic, otherwise False
        res : int, default=101
            number of grid points in each dimension for regridding from FEM mesh
        linear_interp : bool, default=False
            whether to use linear interpolation (True) or cubic spline (False) for
            temporal interpolation of the fluid data.
        vel_conv : float, optional
            scalar to multiply the velocity by in order to convert units to 
            match the spatial grid units
        '''

        ##### Parse parameters and read in data #####
        self.path = path
        self.vel_conv = vel_conv

        path = Path(path)
        if not path.is_dir(): 
            raise FileNotFoundError("Directory {} not found!".format(str(path)))

        ### Load fluid data ###
        print('Reading vtu fluid data...')
        flow, mesh, flow_times = self._read_vtufile(self.path, res=res)
        print('Done!')

        # shift domain to quadrant 1
        flow_points = (mesh[0]-mesh[0][0], mesh[1]-mesh[1][0],
                        mesh[2]-mesh[2][0])
        fluid_domain_LLC = (mesh[0][0], mesh[1][0], mesh[2][0])
        # It is assumed that the fluid spatial grid includes all 
        # domain boundaries.
        self.L = [flow_points[0][-1], flow_points[1][-1], flow_points[2][-1]]

        if self.vel_conv is not None:
            print("Converting vel units by a factor of {}.".format(self.vel_conv))
            for ii, d in enumerate(flow):
                flow[ii] = d*self.vel_conv
        
        if not linear_interp:
            linear_interp = None
        super().__init__(flow, flow_points, flow_times, linear_interp, periodic_dim,
                         fluid_domain_LLC=fluid_domain_LLC)
        


    def _read_vtufile(self, path, res=101):
        '''Reads in one vtu file with fluid velocity data on the FEM mesh at 
        multiple time points exported from COMSOL Multiphysics. Regrids this 
        data onto a regular grid with resolution res in each dimension.

        Parameters
        ----------
        path : string
            path to vtu data, incl. file extension
        res : int, default=101
            number of grid points in each dimension for regridding

        Returns
        -------
        flow : list of ndarray (fluid data)
        mesh : list of 1D arrays of grid points in x, y, and z directions
        flow_times : ndarray of times at which the fluid velocity is specified
        '''

        ### Gather data ###
        flow = [[], [], []]
        flow_times = []

        points, data, data_names = _dataio.read_vtu_Unstructured_Grid_Points_FEM(path)

        # Create regular grid
        x = np.linspace(points[:,0].min(), points[:,0].max(), res)
        y = np.linspace(points[:,1].min(), points[:,1].max(), res)
        if points[:,2].max() == points[:,2].min():
            # 2D case
            z_dir = False
            point_list = np.array([[px,py] for px in x for py in y])
            points = points[:,:2]
            mesh = [x, y]
        else:
            z = np.linspace(points[:,2].min(), points[:,2].max(), res)
            z_dir = True
            point_list = np.array([[px,py,pz] for px in x for py in y for pz in z])
            mesh = [x, y, z]

        # In COMSOL, each three arrays correspond to the x, y, z components of velocity
        #   at each mesh point, with each array being length N where N is the
        #   number of mesh points. Each set of three is a different time point.
        #   So total number of time points is data_size/3.
        for n in range(0, len(data), 3):
            interp = interpolate.LinearNDInterpolator(points, data[n])
            flow[0].append(interp(point_list))
            interp = interpolate.LinearNDInterpolator(points, data[n+1])
            flow[1].append(interp(point_list))
            if z_dir:
                interp = interpolate.LinearNDInterpolator(points, data[n+2])
                flow[2].append(interp(point_list).reshape((res,res,res)))
                flow[0][-1] = flow[0][-1].reshape((res,res,res))
                flow[1][-1] = flow[1][-1].reshape((res,res,res))
            else:
                flow[0][-1] = flow[0][-1].reshape((res,res))
                flow[1][-1] = flow[1][-1].reshape((res,res))
            # get flow time
            flow_times.append(float(data_names[n][data_names[n].rfind('t=')+2:]))

        flow = [np.array(flow[0]).squeeze(), np.array(flow[1]).squeeze(),
                np.array(flow[2]).squeeze()]

        return flow, mesh, flow_times



class OpenFOAMData(FluidData):

    def __init__(self, path, INUM=None, periodic_dim=(False, False, False),
                 vel_conv=None, require_boundary=True):
        '''Reads finite-volume fluid velocity data from an OpenFOAM VTK export
        (the output of ``foamToVTK``) and creates a FluidData instance from it.

        The export is a ``.vtm.series`` JSON index naming one ``.vtm`` manifest
        per timestep, each of which names an ``internal.vtu`` holding the volume
        data plus one ``.vtp`` per boundary patch. Times are read from the series
        index and shifted so that the first dump loaded is at a Planktos
        environment time of 0.0.

        Two properties of such an export need handling, and both are done once at
        construction because the mesh never moves:

        **The data is cell data on a cell-center lattice.** OpenFOAM is
        finite-volume, so ``U`` is specified per cell, and the cells arrive in no
        particular order -- reshaping them without reordering scrambles the field.
        The lattice and the reordering permutation are recovered by
        ``_build_lattice``.

        **The cell centers are inset half a cell from the domain.** Taken alone
        they would report a domain one cell narrower than the real one in every
        direction. The boundary patches close that gap exactly, and are spliced
        onto the six faces of the interior block to recover the true extent.

        Note that the resulting grid is rectilinear but *not* uniform: the two
        outermost intervals in each direction are half-width, being the distance
        from the outermost cell centers to the domain boundary.

        Despite the ``vtkUnstructuredGrid`` container, the mesh is required to be
        rectilinear -- Planktos interpolates on a tensor-product grid. That is
        verified rather than assumed; see ``_build_lattice``.

        A dump that the series index declares but which is not on disk is skipped
        with a warning, and the timeline is built densely over the dumps that do
        exist. A truncated or interrupted export is an ordinary thing to be
        handed, and discovering the gap partway through a long streaming run
        would be the worst possible moment for it.

        If INUM (interval number) is set to an integer >= 4, the data will be
        dynamically loaded as needed with INUM intervals between the temporal
        data sets available at any given time.

        Parameters
        ----------
        path : string
            path to the directory holding the ``.vtm.series`` index (typically
            the ``VTK`` directory foamToVTK writes), or to the index file itself
        INUM : int > 3, True, or None (default)
            max number of linearly splined intervals at any one time. If True,
            all time-varying fluid data is linearly splined at once. If None, all
            is cubically splined instead. Note the number of time points needed
            is 1+INUM.
        periodic_dim : list of 3 bool, default=(False, False, False)
            True if that spatial dimension is periodic, otherwise False. Defaults
            to non-periodic throughout, since a finite-volume export of this shape
            is bounded by patches rather than by periodic images.
        vel_conv : float, optional
            scalar to multiply the velocity by in order to convert units to
            match the spatial grid units
        require_boundary : bool, default=True
            whether to require that every one of the six domain faces is covered
            by a boundary patch. If a face is missing, the domain would be short
            by half a cell in that direction, shifting every coordinate in it
            with nothing downstream able to detect the error, so this raises by
            default. False is not yet implemented; the intended behavior is to
            regrid the cell-centered data out to the domain edges and warn.
        '''

        ##### Parse parameters #####
        self.path = path
        self.vel_conv = vel_conv
        self.require_boundary = require_boundary
        # The boundary-condition corner warning is a property of the dataset,
        # not of the timestep, so say it once rather than on every window slide.
        self._warned_bc_corner = False
        if INUM is not None and INUM is not True:
            assert INUM > 3, 'INUM must be at least 4.'

        ##### Resolve the series index and the dumps that actually exist #####
        self._dumps, flow_times = self._read_series(path)
        # Dump "numbers" are a dense 0-based index over the dumps that exist, NOT
        # the numbers in the directory names (787, 800, ... 1034, which are
        # neither consecutive nor gap-free). FluidData.update_spline does index
        # arithmetic on d_start/d_finish -- d_start = loaded_dump_bnds[1]+1, and
        # so on -- so dump numbers must step by one in lockstep with flow_times.
        # Indexing the surviving series makes that true by construction, and
        # keeps load_dumpfiles from ever being handed a filename that is absent.
        self.d_start = 0
        self.d_finish = len(self._dumps) - 1

        ##### Establish the grid, once: the mesh does not move #####
        print('Reading OpenFOAM mesh...')
        self._build_grid(self._dumps[0])

        ##### Load fluid data #####
        print('Reading OpenFOAM fluid data...')
        if flow_times is None:
            # Single dump: time-invariant flow.
            flow = [f[0] for f in self.load_dumpfiles(0, 0)]
        elif INUM is None or INUM is True:
            flow = self.load_dumpfiles(self.d_start, self.d_finish)
        else:
            flow = self.load_dumpfiles(self.d_start, self.d_start+INUM)
            # record the inclusive bounds of the starting dump numbers to be used
            self.loaded_dump_bnds = (self.d_start, self.d_start+INUM)
            # same, but based off of zero to correspond with flow_times indices
            self.loaded_idx_bnds = (0, INUM)
        print('Done!')

        # shift domain to quadrant 1
        mesh = self._grid
        flow_points = tuple(m - m[0] for m in mesh)
        fluid_domain_LLC = tuple(m[0] for m in mesh)
        # The boundary splice above is what makes this true: the spatial grid
        # includes all domain boundaries.
        self.L = [fp[-1] for fp in flow_points]

        super().__init__(flow, flow_points, flow_times, INUM, periodic_dim,
                         fluid_domain_LLC=fluid_domain_LLC)



    def load_dumpfiles(self, d_start, d_finish):
        '''
        Dynamically load additional OpenFOAM data.

        d_start and d_finish are inclusive indices into the series of dumps that
        exist on disk, not the dump numbers in the directory names.
        '''
        if d_finish < d_start:
            # An empty range. Reachable only from a degenerate slide past the
            # end of the dataset; return correctly shaped empties rather than
            # np.array([]), whose (0,) shape would fail to concatenate against
            # the window with a message naming neither the range nor the cause.
            # The array-slicing loaders get this shape for free from their
            # slices; this one builds a list, so it has to be explicit.
            empty = np.zeros((0, *(n+2 for n in self._shape)))
            return [empty.copy() for _ in range(3)]

        flow = [[], [], []]
        for n in range(d_start, d_finish+1):
            vel = self._read_dump(self._dumps[n])
            for dim in range(3):
                flow[dim].append(vel[..., dim])
        flow = [np.array(f) for f in flow]

        if self.vel_conv is not None:
            for ii, d in enumerate(flow):
                flow[ii] = d*self.vel_conv
        return flow



    def _read_series(self, path):
        '''Resolve the series index and work out which of its dumps are present.

        Returns
        -------
        dumps : list of dict
            the resolved {dataset name: path} manifest of each dump that exists,
            in time order
        flow_times : ndarray of floats, or None if there is only one dump
            time of each entry of dumps, shifted so the first is 0.0
        '''

        path = Path(path)
        if path.is_dir():
            found = sorted(path.glob('*.vtm.series'))
            if len(found) == 0:
                raise FileNotFoundError(
                    "No .vtm.series index found in {}. ".format(str(path))+
                    "Expected the VTK directory written by foamToVTK.")
            if len(found) > 1:
                raise RuntimeError(
                    "Found {} .vtm.series indices in {}: {}. Pass the one to "
                    "read.".format(len(found), str(path),
                                   [f.name for f in found]))
            series = found[0]
        elif path.is_file():
            series = path
        else:
            raise FileNotFoundError("{} not found!".format(str(path)))
        self.series_path = series

        files, times = _dataio.read_vtm_series(series)

        ##### Resolve which dumps actually exist, eagerly #####
        # Eagerly, at construction, rather than at the window slide that needs a
        # missing file: under dynamic loading that raise lands arbitrarily deep
        # into a run, which is the worst possible moment and exactly what
        # streaming makes likely.
        def _fmt(t):
            return 'unknown' if t is None or np.isnan(t) else '{:g}'.format(t)

        dumps = []; keep_times = []; missing = []
        for f, t in zip(files, times):
            if not f.is_file():
                missing.append(_fmt(t))
                continue
            datasets, mtime = _dataio.read_vtm_manifest(f)
            if 'internal' not in datasets:
                raise RuntimeError(
                    "Manifest {} names no 'internal' dataset. Expected the "
                    "volume data written by foamToVTK.".format(f.name))
            # The series index is the primary time source; fall back to the
            # manifest's own TimeValue for an entry it failed to describe.
            dtime = mtime if np.isnan(t) else t
            if not datasets['internal'].is_file():
                missing.append(_fmt(dtime))
                continue
            dumps.append(datasets)
            keep_times.append(dtime)

        if len(dumps) == 0:
            raise FileNotFoundError(
                "None of the {} dumps declared by {} are present on "
                "disk.".format(len(files), series.name))

        if len(missing) > 0:
            # Warn rather than fail: a truncated or interrupted export is normal.
            # But warn loudly -- silence here would be worse than the failure,
            # since the run would complete with nothing indicating that the
            # timeline is not the one the index declared.
            warnings.warn(
                "{} of {} dumps declared by {} are not on disk and have been "
                "skipped: t = {}. The timeline is built over the {} dumps that "
                "remain.".format(len(missing), len(files), series.name,
                                 ', '.join(missing), len(dumps)), UserWarning)

        if None in keep_times:
            raise RuntimeError(
                "No time information for at least one dump in {}: the series "
                "index gave none and the manifest carries no TimeValue.".format(
                    series.name))

        if len(dumps) == 1:
            return dumps, None

        flow_times = np.array(keep_times, dtype=float)
        # shift so that the first dump loaded corresponds to environment time 0
        flow_times = flow_times - flow_times[0]

        ##### Warn about non-uniform spacing left behind by any gaps #####
        # Interpolation error scales with the dump interval, so a series with a
        # hole is measurably worse across that hole and the user should be told
        # where, separately from being told which dumps are absent.
        dt = np.diff(flow_times)
        if not np.allclose(dt, dt[0], rtol=1e-6):
            wide = np.nonzero(dt > dt.min()*(1+1e-6))[0]
            warnings.warn(
                "Dump times are not evenly spaced: intervals range from {:g} to "
                "{:g}. Temporal interpolation is less accurate across the wider "
                "ones, which begin at t = {}.".format(
                    dt.min(), dt.max(),
                    ', '.join('{:g}'.format(flow_times[i]) for i in wide)),
                UserWarning)

        return dumps, flow_times



    @staticmethod
    def _cluster_axis(v, rel_tol=1e-5):
        '''Group the values of one coordinate into the levels of a grid axis.

        The cells of an unstructured export arrive in no particular order, so the
        grid has to be recovered from the coordinates themselves: this returns
        the sorted coordinate of each level and the level index of each cell.

        Grouping, not averaging, is the point. The coordinates are exact -- what
        is not available is the knowledge of which cells share a level. Sorting
        and splitting at gaps supplies it. A level's coordinate is then taken as
        the mean of its members, but any member would do: measured on a real
        775k-cell export, a level held at most 8 distinct float64 values spanning
        7.6e-19, which is 5e-16 of a cell width, and mean/first/min all reproduce
        the lattice of an independently-written boundary patch bit-for-bit.

        That tiny spread is why np.unique cannot be used for this. It reported 79
        levels where 66 existed on that same data -- the values are right, but
        cells that share a level do not always land on the same float64.

        rel_tol is a fraction of the axis' full span, and is not delicate: it
        needs only to sit between the roundoff spread and the true grid spacing.
        On the real data any value from 1e-3 to 1e-8 gave identical results.
        '''

        order = np.argsort(v, kind='stable')
        s = v[order]
        tol = (s[-1] - s[0])*rel_tol
        brk = np.nonzero(np.diff(s) > tol)[0]
        starts = np.concatenate(([0], brk+1))
        ends = np.concatenate((brk+1, [len(s)]))

        vals = np.array([s[a:b].mean() for a, b in zip(starts, ends)])
        lvl = np.empty(len(v), dtype=np.int64)
        for k, (a, b) in enumerate(zip(starts, ends)):
            lvl[order[a:b]] = k
        return vals, lvl



    def _build_lattice(self, centers, axes=(0, 1, 2)):
        '''Recover a rectilinear grid, and the cell ordering, from cell centers.

        Parameters
        ----------
        centers : Nx3 ndarray
            cell center coordinates, in the order the file stores them
        axes : tuple of int
            which spatial axes the cells vary over. Two axes for a boundary
            patch, which is planar in the third.

        Returns
        -------
        grid : list of 1D ndarray
            coordinates along each axis in axes
        perm : ndarray of int
            ordering such that ``field[perm].reshape(shape)`` is indexed by the
            axes in order, each increasing with its coordinate
        shape : tuple of int

        Raises
        ------
        ValueError
            if the cells do not form a complete tensor-product grid. This is the
            check that Planktos' rectilinear-grid assumption actually holds for
            the dataset in hand -- an unstructured container says nothing about
            the mesh inside it, so it is verified rather than assumed. Requiring
            the linear index to be a permutation of arange is total: it fails on
            a missing cell, a duplicated one, refinement, or any non-tensor mesh.
        '''

        grid = []; idx = []
        for d in axes:
            vals, lvl = self._cluster_axis(centers[:, d])
            grid.append(vals); idx.append(lvl)
        shape = tuple(len(g) for g in grid)

        lin = np.zeros(len(centers), dtype=np.int64)
        for k in range(len(axes)):
            lin = lin*shape[k] + idx[k]

        total = int(np.prod(shape))
        if len(centers) != total or \
                not np.array_equal(np.sort(lin), np.arange(total)):
            raise ValueError(
                "Fluid data is not on a rectilinear grid: {} cells do not form "
                "a complete {} lattice. Planktos interpolates on a "
                "tensor-product grid, so a refined or genuinely unstructured "
                "mesh must be resampled before it can be read.".format(
                    len(centers), ' x '.join(str(s) for s in shape)))

        return grid, np.argsort(lin), shape



    def _build_grid(self, datasets):
        '''Establish the spatial grid and every reordering the loader will need.

        Done once, from one dump, because the mesh does not move -- the point
        coordinates of an OpenFOAM export are written redundantly into every
        timestep but are bit-identical across them. Only field arrays are read
        thereafter.

        Sets ``_grid`` (the assembled coordinate arrays, boundary included),
        ``_perm``/``_shape`` for the interior, and ``_faces``, which says where
        each of the six domain faces gets its data.
        '''

        ##### Interior #####
        centers, _, _ = _dataio.read_vtkxml_cell_data(datasets['internal'],
                                                      arrays=())
        interior, self._perm, self._shape = self._build_lattice(centers)

        ##### Boundary patches #####
        # Faces are identified by geometry, not by patch name. A patch file may
        # hold several faces (foamToVTK writes all four lateral walls into one
        # walls.vtp), and names are case-specific -- inlet/outlet/walls here,
        # something else in the next export. Which face a cell belongs to is
        # decided by which interior axis range it falls outside of.
        self._faces = {}
        for name, fname in datasets.items():
            if name == 'internal':
                continue
            pcenters, _, _ = _dataio.read_vtkxml_cell_data(fname, arrays=())
            # For each axis, is this cell outside the interior cell-center range?
            outside = np.stack(
                [np.where(pcenters[:, d] < interior[d][0], 0,
                          np.where(pcenters[:, d] > interior[d][-1], 1, -1))
                 for d in range(3)], axis=1)
            n_out = (outside >= 0).sum(axis=1)
            if np.any(n_out != 1):
                raise RuntimeError(
                    "Boundary patch '{}' has {} cells that do not lie on "
                    "exactly one domain face. A patch is expected to be planar "
                    "and to sit just outside the interior cell centers.".format(
                        name, int((n_out != 1).sum())))
            for d in range(3):
                for side in (0, 1):
                    sel = np.nonzero(outside[:, d] == side)[0]
                    if len(sel) == 0:
                        continue
                    tan = tuple(a for a in range(3) if a != d)
                    fgrid, fperm, fshape = self._build_lattice(pcenters[sel],
                                                               axes=tan)
                    # The patch has to sit on the interior's own in-plane
                    # lattice, or it could not be spliced on without resampling.
                    for k, a in enumerate(tan):
                        if len(fgrid[k]) != len(interior[a]) or \
                                not np.allclose(fgrid[k], interior[a]):
                            raise RuntimeError(
                                "Boundary patch '{}' does not share the "
                                "interior grid in the {} direction, so it "
                                "cannot be spliced on.".format(
                                    name, 'xyz'[a]))
                    plane = float(pcenters[sel, d].mean())
                    if (d, side) in self._faces:
                        raise RuntimeError(
                            "Two boundary patches cover the {} face of the "
                            "domain.".format('xyz'[d]+'-+'[side]))
                    # Keyed by dataset NAME, not by the path it resolved to in
                    # this dump: the patch geometry is fixed but its data varies
                    # in time, so each timestep must read its own file. sel[fperm]
                    # is a single index array doing selection and reordering at
                    # once, so a timestep's read is one fancy-index and a reshape.
                    self._faces[(d, side)] = (name, sel[fperm], fshape, plane)

        missing = [(d, side) for d in range(3) for side in (0, 1)
                   if (d, side) not in self._faces]
        if len(missing) > 0:
            names = ', '.join('{}{}'.format('xyz'[d], '-+'[s])
                              for d, s in missing)
            if self.require_boundary:
                raise RuntimeError(
                    "No boundary patch covers the {} face(s) of the "
                    "domain. Cell centers are inset half a cell, so without "
                    "them the domain would be reported short by half a cell "
                    "there, shifting every coordinate in it with nothing "
                    "downstream able to detect it. Pass require_boundary=False "
                    "once regridding to the domain edge is implemented.".format(
                        names))
            raise NotImplementedError(
                "require_boundary=False is not implemented yet. Recovering the "
                "domain edge without a boundary patch means regridding the "
                "cell-centered data outward, which is planned but not built; "
                "the {} face(s) have no patch.".format(names))

        ##### Assemble the grid coordinates #####
        self._grid = []
        for d in range(3):
            self._grid.append(np.concatenate((
                [self._faces[(d, 0)][3]], interior[d],
                [self._faces[(d, 1)][3]])))
            if not np.all(np.diff(self._grid[-1]) > 0):
                raise RuntimeError(
                    "Boundary patches for the {} direction do not lie outside "
                    "the interior cell centers.".format('xyz'[d]))



    def _read_dump(self, datasets, atol=1e-12):
        '''Read one timestep and assemble it onto the full domain grid.

        Returns an (nx+2, ny+2, nz+2, 3) array: the interior block surrounded by
        a shell of boundary values.

        The shell is filled in three stages, because the patch files cover the
        six faces but not the lines and points where faces meet. An edge cell of
        the assembled grid lies on two faces at once and appears in neither
        patch; it is filled from the two faces that meet there, and a corner from
        the three edges. Where the two sides disagree the fill is a compromise
        and says so -- that is a genuine discontinuity in the boundary
        conditions, e.g. an inlet running into a no-slip wall.
        '''

        nx, ny, nz = self._shape
        vel = np.zeros((nx+2, ny+2, nz+2, 3))

        _, data, _ = _dataio.read_vtkxml_cell_data(
            datasets['internal'], arrays=('U',), load_cell_coordinates=False)
        U = data['U']
        if len(U) != nx*ny*nz:
            raise RuntimeError(
                "{} has {} cells; the mesh established from the first dump has "
                "{}. The mesh is assumed not to change across the "
                "series.".format(datasets['internal'].name, len(U), nx*ny*nz))
        vel[1:-1, 1:-1, 1:-1, :] = U[self._perm].reshape(nx, ny, nz, 3)

        ##### Stage 1: the six faces, from the patch files #####
        patches = {}
        for (d, side), (name, sel, fshape, _) in self._faces.items():
            if name not in patches:
                if name not in datasets:
                    raise RuntimeError(
                        "Dump {} has no '{}' boundary patch, but the mesh "
                        "established from the first dump needs it for the {} "
                        "face.".format(datasets['internal'].parent.name, name,
                                       'xyz'[d]+'-+'[side]))
                _, pdata, _ = _dataio.read_vtkxml_cell_data(
                    datasets[name], arrays=('U',), load_cell_coordinates=False)
                patches[name] = pdata['U']
            face = patches[name][sel].reshape(*fshape, 3)
            idx = [slice(1, -1)]*3
            idx[d] = 0 if side == 0 else -1
            vel[tuple(idx)] = face

        ##### Stage 2: the twelve edges, from the two faces meeting there #####
        # Each edge runs along axis c with axes a and b at their extremes. Only
        # the interior run of it is defined by the faces; the two ends are
        # corners, where three faces meet, and are left to stage 3.
        disagreement = 0.
        for a in range(3):
            for b in range(a+1, 3):
                for sa in (0, -1):
                    for sb in (0, -1):
                        idx = [slice(1, -1)]*3
                        idx[a] = sa; idx[b] = sb
                        # Step one cell inward along a and the point lands on
                        # face b; step inward along b and it lands on face a.
                        # Both were filled in stage 1.
                        from_b = list(idx); from_b[a] = 1 if sa == 0 else -2
                        from_a = list(idx); from_a[b] = 1 if sb == 0 else -2
                        va = vel[tuple(from_a)]
                        vb = vel[tuple(from_b)]
                        vel[tuple(idx)] = 0.5*(va + vb)
                        disagreement = max(disagreement,
                                           float(np.abs(va-vb).max()))

        ##### Stage 3: the eight corners, from the three edges meeting there ###
        for sx in (0, -1):
            for sy in (0, -1):
                for sz in (0, -1):
                    ix = 1 if sx == 0 else -2
                    iy = 1 if sy == 0 else -2
                    iz = 1 if sz == 0 else -2
                    vel[sx, sy, sz, :] = (vel[ix, sy, sz, :] +
                                          vel[sx, iy, sz, :] +
                                          vel[sx, sy, iz, :])/3

        if disagreement > atol and not self._warned_bc_corner:
            self._warned_bc_corner = True
            warnings.warn(
                "Boundary patches disagree by up to {:g} where they meet along "
                "the edges of the domain; those cells have been filled with the "
                "average of the two faces. This is a discontinuity in the "
                "boundary conditions themselves (an inflow meeting a no-slip "
                "wall, say), not an error in the data.".format(disagreement),
                UserWarning)

        return vel



######## Legacy function for regridding fluid velocity data ########
# This was in the Environment class, but is no longer used.
# It's here in case we need to port it later.
# Also, it is not robust to rectilinear grids.

# def center_cell_regrid(self):
#     '''Re-grids data that was specified at the center of cells instead of
#     at the corners.

#     NOTE! This needs to be called *before* any immersed meshes are loaded.
#     It will NOT look for and properly shift these meshes.
    
#     Software has a tendency to output data files where the fluid mesh is 
#     specified at the center of cells rather than at the corners. This will 
#     be readily apparent if Planktos loads your fluid velocity data and 
#     reports spatial dimensions one dx, dy, and dz smaller than you were
#     expecting. To fix this, Planktos will interpolate/extrapolate the fluid 
#     velocity mesh using the default method to get additional grid points on 
#     the edge of the domain.

#     Periodicity can be enforced in specified dimensions.

#     Parameters
#     ----------
#     periodic_dim : list-like of 2 or 3 bool, default=(True, True, False)
#         True if that spatial dimension is periodic, otherwise False.
#         The 3rd entry will be ignored in the 2D case.
#     '''
    
#     fpoints = self.flow.flow_points

#     # Detect cell width in each dimension based on the first two coordinates 
#     #   in each spatial dimension
#     dx = fpoints[0][1] - fpoints[0][0]
#     dy = fpoints[1][1] - fpoints[1][0]
#     if len(self.L) > 2:
#         dz = fpoints[2][1] - fpoints[2][0]
#         DIM3 = True
#     else:
#         DIM3 = False

#     ### Create a list of positions at which we need to extrapolate the ###
#     ###   velocity field                                               ###
#     x_ends = [-dx/2, fpoints[0][-1]+dx/2]
#     y_ends = [-dy/2, fpoints[1][-1]+dy/2]
#     bndry_list = []
#     if not DIM3:
#         # edges
#         bndry_list += [[x, y_ends[0]] for x in fpoints[0]]
#         bndry_list += [[x, y_ends[1]] for x in fpoints[0]]
#         bndry_list += [[x_ends[0], y] for y in fpoints[1]]
#         bndry_list += [[x_ends[1], y] for y in fpoints[1]]
#         # points
#         bndry_list += [[x_ends[0],y_ends[0]],[x_ends[0],y_ends[1]],
#                         [x_ends[1],y_ends[0]],[x_ends[1],y_ends[1]]]
#     else:
#         z_ends = [-dz/2, fpoints[2][-1]+dz/2]
#         # sides
#         bndry_list += [[x,y,z_ends[0]] for x in fpoints[0] for y in fpoints[1]]
#         bndry_list += [[x,y,z_ends[1]] for x in fpoints[0] for y in fpoints[1]]
#         bndry_list += [[x,y_ends[0],z] for x in fpoints[0] for z in fpoints[2]]
#         bndry_list += [[x,y_ends[1],z] for x in fpoints[0] for z in fpoints[2]]
#         bndry_list += [[x_ends[0],y,z] for y in fpoints[1] for z in fpoints[2]]
#         bndry_list += [[x_ends[1],y,z] for y in fpoints[1] for z in fpoints[2]]
#         # edges
#         bndry_list += [[x, y_ends[0], z_ends[0]] for x in fpoints[0]]
#         bndry_list += [[x, y_ends[0], z_ends[1]] for x in fpoints[0]]
#         bndry_list += [[x, y_ends[1], z_ends[0]] for x in fpoints[0]]
#         bndry_list += [[x, y_ends[1], z_ends[1]] for x in fpoints[0]]
#         bndry_list += [[x_ends[0], y, z_ends[0]] for y in fpoints[1]]
#         bndry_list += [[x_ends[0], y, z_ends[1]] for y in fpoints[1]]
#         bndry_list += [[x_ends[1], y, z_ends[0]] for y in fpoints[1]]
#         bndry_list += [[x_ends[1], y, z_ends[1]] for y in fpoints[1]]
#         bndry_list += [[x_ends[0], y_ends[0], z] for z in fpoints[2]]
#         bndry_list += [[x_ends[0], y_ends[1], z] for z in fpoints[2]]
#         bndry_list += [[x_ends[1], y_ends[0], z] for z in fpoints[2]]
#         bndry_list += [[x_ends[1], y_ends[1], z] for z in fpoints[2]]
#         # points
#         bndry_list += [[x_ends[0],y_ends[0],z_ends[0]],
#                         [x_ends[0],y_ends[0],z_ends[1]],
#                         [x_ends[0],y_ends[1],z_ends[0]],
#                         [x_ends[1],y_ends[0],z_ends[1]],
#                         [x_ends[0],y_ends[1],z_ends[1]],
#                         [x_ends[1],y_ends[0],z_ends[1]],
#                         [x_ends[1],y_ends[1],z_ends[0]],
#                         [x_ends[1],y_ends[1],z_ends[1]]]

#     ### Include periodicity, if applicable, by extending out the fluid field ###
#     flowshape = np.array(self.flow.fshape)
#     idx = []
#     if len(self.flow.fshape) == len(self.L):
#         # non time-varying flow
#         startdim = 0
#     else:
#         startdim = 1
#     for ii in range(2):
#         if self.flow.periodic_dim[ii]:
#             flowshape[startdim+ii] += 2
#             idx.append([1,flowshape[startdim+ii]-1])
#         else:
#             idx.append([0,flowshape[startdim+ii]])
#     if DIM3:
#         if self.flow.periodic_dim[2]:
#             flowshape[startdim+2] += 2
#             idx.append([1,flowshape[startdim+2]-1])
#         else:
#             idx.append([0,flowshape[startdim+2]])
    
#     if DIM3:
#         flow = [np.zeros(flowshape) for ii in range(3)]
#         flow_points = []
#         for ii in range(3):
#             flow[ii][...,idx[0][0]:idx[0][1],idx[1][0]:idx[1][1],idx[2][0]:idx[2][1]] = self.flow[ii]
#         if self.flow.periodic_dim[0]:
#             for ii in range(3):
#                 flow[ii][...,0,:,:] = flow[ii][...,-2,:,:]
#                 flow[ii][...,-1,:,:] = flow[ii][...,1,:,:]
#             flow_points.append(np.insert(fpoints[0], # what
#                                             [0,len(fpoints[0])], # loc
#                                             [-dx,fpoints[0][-1]+dx])) # vals
#         else:
#             flow_points.append(fpoints[0])
#         if self.flow.periodic_dim[1]:
#             for ii in range(3):
#                 flow[ii][...,0,:] = flow[ii][...,-2,:]
#                 flow[ii][...,-1,:] = flow[ii][...,1,:]
#             flow_points.append(np.insert(fpoints[1],
#                                             [0,len(fpoints[1])],
#                                             [-dy,fpoints[1][-1]+dy]))
#         else:
#             flow_points.append(fpoints[1])
#         if self.flow.periodic_dim[2]:
#             for ii in range(3):
#                 flow[ii][...,0] = flow[ii][...,-2]
#                 flow[ii][...,-1] = flow[ii][...,1]
#             flow_points.append(np.insert(fpoints[2],
#                                             [0,len(fpoints[2])],
#                                             [-dz,fpoints[2][-1]+dz]))
#         else:
#             flow_points.append(fpoints[2])
#     else:
#         flow = [np.zeros(flowshape) for ii in range(2)]
#         flow_points = []
#         for ii in range(2):
#             flow[ii][...,idx[0][0]:idx[0][1],idx[1][0]:idx[1][1]] = self.flow[ii]
#         if self.flow.periodic_dim[0]:
#             for ii in range(2):
#                 flow[ii][...,0,:] = flow[ii][...,-2,:]
#                 flow[ii][...,-1,:] = flow[ii][...,1,:]
#             flow_points.append(np.insert(fpoints[0], # what
#                                             [0,len(fpoints[0])], # loc
#                                             [-dx,fpoints[0][-1]+dx])) # vals
#         else:
#             flow_points.append(fpoints[0])
#         if self.flow.periodic_dim[1]:
#             for ii in range(2):
#                 flow[ii][...,0] = flow[ii][...,-2]
#                 flow[ii][...,-1] = flow[ii][...,1]
#             flow_points.append(np.insert(fpoints[1],
#                                             [0,len(fpoints[1])],
#                                             [-dy,fpoints[1][-1]+dy]))
#         else:
#             flow_points.append(fpoints[1])

#     ### Interpolate the new points ###
#     if startdim == 0:
#         # non time-varying flow
#         new_vecs = self.interpolate_flow(bndry_list, flow, flow_points)
#     else:
#         new_vecs = []
#         for t_idx in range(self.flow.fshape[0]):
#             this_flow = [flow[ii][t_idx,...] for ii in range(len(flow))]
#             new_vecs.append(self.interpolate_flow(bndry_list, this_flow, flow_points))

#     ### Incorporate the new points into the fluid field and mesh ###
#     if DIM3:
#         intervals = [dx,dy,dz]
#     else:
#         intervals = [dx,dy]
#     flow_points = [np.insert(fpoints[ii]+interval/2,
#                                 [0,len(fpoints[ii])],
#                                 [0,fpoints[ii][-1]+interval])
#                                 for ii,interval in enumerate(intervals)]
#     flowshape = np.array(self.flow.fshape)
#     if startdim == 0:
#         flowshape += 2
#     else:
#         flowshape[1:] += 2
#     shp = [len(points) for points in fpoints]

#     def bndry_add3d(fshape, shp, this_vecs):
#         f = [np.zeros(fshape) for ii in range(3)]
#         for dim in range(3):
#             # sides
#             f[dim][1:-1,1:-1,0] = np.reshape(this_vecs[:shp[0]*shp[1],dim],(shp[0],shp[1]))
#             s = shp[0]*shp[1]
#             f[dim][1:-1,1:-1,-1] = np.reshape(this_vecs[s:s+shp[0]*shp[1],dim],(shp[0],shp[1]))
#             s += shp[0]*shp[1]
#             f[dim][1:-1,0,1:-1] = np.reshape(this_vecs[s:s+shp[0]*shp[2],dim],(shp[0],shp[2]))
#             s += shp[0]*shp[2]
#             f[dim][1:-1,-1,1:-1] = np.reshape(this_vecs[s:s+shp[0]*shp[2],dim],(shp[0],shp[2]))
#             s += shp[0]*shp[2]
#             f[dim][0,1:-1,1:-1] = np.reshape(this_vecs[s:s+shp[1]*shp[2],dim],(shp[1],shp[2]))
#             s += shp[1]*shp[2]
#             f[dim][-1,1:-1,1:-1] = np.reshape(this_vecs[s:s+shp[1]*shp[2],dim],(shp[1],shp[2]))
#             s += shp[1]*shp[2]
#             # edges
#             f[dim][1:-1,0,0] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
#             f[dim][1:-1,0,-1] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
#             f[dim][1:-1,-1,0] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
#             f[dim][1:-1,-1,-1] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
#             f[dim][0,1:-1,0] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
#             f[dim][0,1:-1,-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
#             f[dim][-1,1:-1,0] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
#             f[dim][-1,1:-1,-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
#             f[dim][0,0,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
#             f[dim][0,-1,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
#             f[dim][-1,0,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
#             f[dim][-1,-1,1:-1] = this_vecs[s:s+shp[2],dim]; s+=shp[2]
#             # points
#             f[dim][0,0,0] = this_vecs[s,dim]
#             f[dim][0,0,-1] = this_vecs[s+1,dim]
#             f[dim][0,-1,0] = this_vecs[s+2,dim]
#             f[dim][-1,0,0] = this_vecs[s+3,dim]
#             f[dim][0,-1,-1] = this_vecs[s+4,dim]
#             f[dim][-1,0,-1] = this_vecs[s+5,dim]
#             f[dim][-1,-1,0] = this_vecs[s+6,dim]
#             f[dim][-1,-1,-1] = this_vecs[s+7,dim]
#         return f

#     def bndry_add2d(fshape, shp, this_vecs):
#         f = [np.zeros(fshape) for ii in range(2)]
#         for dim in range(2):
#             s=0
#             # edges
#             f[dim][1:-1,0] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
#             f[dim][1:-1,-1] = this_vecs[s:s+shp[0],dim]; s+=shp[0]
#             f[dim][0,1:-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
#             f[dim][-1,1:-1] = this_vecs[s:s+shp[1],dim]; s+=shp[1]
#             # points
#             f[dim][0,0] = this_vecs[s,dim]
#             f[dim][0,-1] = this_vecs[s+1,dim]
#             f[dim][-1,0] = this_vecs[s+2,dim]
#             f[dim][-1,-1] = this_vecs[s+3,dim]
#         return f

#     if DIM3 and startdim == 0:
#         # time invariant, 3D
#         flow = bndry_add3d(flowshape, shp, new_vecs)
#     elif DIM3 and startdim == 1:
#         flow = [np.zeros(flowshape) for ii in range(3)]
#         for n, this_vecs in enumerate(new_vecs):
#             f = bndry_add3d(flowshape[1:], shp, this_vecs)
#             flow[0][n,...]=f[0]; flow[1][n,...]=f[1]; flow[2][n,...]=f[2]
#     elif not DIM3 and startdim == 0:
#         flow = bndry_add2d(flowshape, shp, new_vecs)
#     else:
#         flow = [np.zeros(flowshape) for ii in range(2)]
#         for n, this_vecs in enumerate(new_vecs):
#             f = bndry_add2d(flowshape[1:], shp, this_vecs)
#             flow[0][n,...]=f[0]; flow[1][n,...]=f[1]

#     ### Add back the original fluid data ###
#     for dim in range(len(flow)):
#         if DIM3:
#             flow[dim][...,1:-1,1:-1,1:-1] = self.flow[dim]
#         else:
#             flow[dim][...,1:-1,1:-1] = self.flow[dim]

#     ### Replace fluid and update domain ###
#     flow_points = tuple(flow_points)
#     fluid_domain_LLC = tuple(np.array(self.flow.fluid_domain_LLC)
#                                 -np.array(intervals)*0.5)
#     self.flow = fluid.FluidData(flow, flow_points, 
#                                 self.flow.flow_times, 
#                                 periodic_dim=self.flow.periodic_dim,
#                                 fluid_domain_LLC=fluid_domain_LLC)
#     self.L = [flow_points[d][-1] for d in range(len(flow_points))]
    
#     self._reset_flow_variables()