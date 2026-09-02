'''Functions and methods for loading and handling fluid data.
These are mainly utilized by the Environment class.

Created: Thurs July 9 2025

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import re
import warnings
import numpy as np
from scipy import interpolate
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



def _wrap_scalar(f, periodic_dim):
    '''Restore the duplicated end line on each periodic axis of a *scalar* field.

    The scalar counterpart of ``_wrap_flow``, which cannot be used here: it takes
    one array per spatial dimension. Derived scalar fields read back from a
    source that omits the periodic endpoint -- IB2d's ``Omega.####.vtk``, for
    instance -- need exactly this.

    Parameters
    ----------
    f : ndarray, indexed [x,y(,z)]
    periodic_dim : sequence of bool, one per axis of f

    Returns
    -------
    ndarray with one extra entry along each periodic axis
    '''

    for d, periodic in enumerate(periodic_dim):
        if periodic:
            f = np.concatenate((f, np.take(f, [0], axis=d)), axis=d)
    return f



def _unwrap_scalar(f, periodic_dim):
    '''Drop the duplicated end line on each periodic axis. Inverse of
    ``_wrap_scalar``, for writing a derived field back in the source's own
    convention.'''

    return f[tuple(slice(0, -1) if periodic else slice(None)
                   for periodic in periodic_dim)]



def _drop_flat_axes(flow, axes, ndim):
    """Remove flat spatial axes from a velocity field, components included.

    The component *along* a flat axis goes with it: Planktos carries one
    velocity component per spatial dimension, so a 2D grid holds a 2D vector.

    Parameters
    ----------
    flow : list of ndarrays
        one per velocity component, each ``([t],i,j,[k])``
    axes : tuple of int
        spatial axes to drop, as :func:`_collapse_flat_axes` identifies them
    ndim : int
        spatial dimension of the data as read, before anything is dropped.
        Used to tell whether the arrays carry a leading time axis.

    Returns
    -------
    list of ndarrays, one shorter and one axis flatter per entry in ``axes``
    """

    if not axes:
        return flow
    # A leading time axis shifts every spatial axis along by one.
    offset = 1 if flow[0].ndim > ndim else 0
    kept = [f for d, f in enumerate(flow) if d not in axes]
    return [np.squeeze(f, axis=tuple(d + offset for d in axes)) for f in kept]



def _collapse_flat_axes(flow, mesh, periodic_dim=None, source=None):
    """Reduce a field written on a one-point-thick grid to its real dimension.

    Everything that describes the grid moves together -- the field, the
    coordinate arrays, the velocity components and the periodicity flags -- so
    that a plane exported as 3D vtk becomes an ordinary 2D Planktos fluid rather
    than a 3D one of zero thickness. IB2d's reader has always done this; this is
    the same rule for the 3D loaders.

    Parameters
    ----------
    flow : list of ndarrays
    mesh : sequence of 1D ndarrays
        coordinates along each axis
    periodic_dim : bool or sequence of bool, optional
    source : str, optional
        named in the warning

    Returns
    -------
    flow, mesh, periodic_dim, axes -- the first three with flat axes removed,
    and the axes that were dropped, for a loader to reapply to later reads

    Raises
    ------
    ValueError
        if fewer than two axes would be left; Planktos is 2D or 3D
    """

    axes = tuple(d for d, c in enumerate(mesh) if len(np.atleast_1d(c)) < 2)
    if not axes:
        return flow, list(mesh), periodic_dim, axes

    names = 'xyz'
    dropped = ', '.join(names[d] for d in axes)
    if len(mesh) - len(axes) < 2:
        raise ValueError(
            "This grid is one point thick along {} of its {} axes, leaving "
            "fewer than the two Planktos needs. {}".format(
                dropped, len(mesh),
                '' if source is None else 'Read from {}.'.format(source)))

    # A component along an axis that is being dropped is discarded with it. That
    #   is free for genuinely planar data, where it is identically zero, and is
    #   a real loss otherwise -- so say which it was.
    lost = [names[d] for d in axes if np.any(flow[d] != 0)]
    warnings.warn(
        "Fluid grid is a single point thick in {}, so it is being read as {}D "
        "data on the remaining axes.{}{}".format(
            dropped, len(mesh) - len(axes),
            '' if source is None else ' Read from {}.'.format(source),
            '' if not lost else
            ' NOTE: the {}-velocity is not everywhere zero and is being '
            'discarded with that axis -- this is a slab of a 3D flow rather '
            'than 2D data.'.format(', '.join(lost))), UserWarning, stacklevel=3)

    flow = _drop_flat_axes(flow, axes, len(mesh))
    mesh = [c for d, c in enumerate(mesh) if d not in axes]
    if periodic_dim is not None and not isinstance(periodic_dim, bool):
        periodic_dim = tuple(v for d, v in enumerate(periodic_dim)
                             if d not in axes)
    return flow, mesh, periodic_dim, axes



def _spline_index(spline, pos):
    """Index a spline at its own knots, as though it were an ``([t],i,j,[k])`` array.

    Shared by both spline classes: they differ in how they interpolate between
    knots, not in what an index means. ``spline.x`` is the knot sequence and
    ``spline(t)`` the field there.

    Parameters
    ----------
    spline : LinearSpline or fCubicSpline
    pos : int, slice, or tuple
        The leading entry indexes time; anything after it indexes into the field
        at that time. Integers may be numpy integers, which is what a caller
        looping over ``np.arange`` or unpacking ``np.searchsorted`` will supply.

    Returns
    -------
    ndarray -- one time point for an integer index, a stack of them for a slice
    """

    knots = spline.x
    lead, rest = (pos[0], pos[1:]) if isinstance(pos, tuple) else (pos, ())

    def at(i):
        field = spline(knots[i])
        return field[rest] if rest else field

    if isinstance(lead, (int, np.integer)):
        return at(lead)
    if isinstance(lead, slice):
        # slice.indices resolves None, negative bounds and negative steps against
        #   the length, which hand-rolled start/stop arithmetic gets wrong: a
        #   negative start such as [-3:] otherwise runs from -3 up to len-1 and
        #   returns len+3 wrapped entries instead of the last three.
        return np.stack([at(i) for i in range(*lead.indices(len(knots)))])
    raise IndexError('Only integers or slices are supported in {}.'.format(
        type(spline).__name__))



def _linear_blend(times, t, get):
    """Linearly interpolate at ``t`` between the two entries bracketing it.

    ``get(i)`` supplies entry ``i``. It is called **once** where ``t`` lands on a
    knot or outside the data and twice otherwise, so a caller whose entries are
    expensive to produce -- a file read, a spline evaluation -- pays only for
    what the interpolation needs. Entries may be arrays of any shape.

    This is the one place the weights of a linear interpolation in time are
    computed, so anything derived per dump and blended here is guaranteed to be
    consistent with the velocity field itself.
    """

    # Bracket t, clamping to the ends so that a time outside the data gets the
    #   constant extrapolation the field itself uses.
    if len(times) < 2 or t <= times[0]:
        return get(0)
    if t >= times[-1]:
        return get(len(times) - 1)
    idx = int(np.searchsorted(times, t)) - 1
    w = (t - times[idx]) / (times[idx+1] - times[idx])
    if w == 0.0:
        # t landed on a knot; one call, and no arithmetic on the nodal values.
        return get(idx)
    lo = get(idx)
    return lo + (get(idx+1) - lo)*w



def _spatial_gradient(f, coords, axis, periodic=False, edge_order=1):
    '''Differentiate along one spatial axis, wrapping if that axis is periodic.

    A periodic axis carries a duplicated end line -- ``FluidData`` requires the
    upper grid edge to repeat the lower one -- so the field genuinely continues
    past either end, and the one-sided difference ``np.gradient`` falls back to
    there is simply wrong. It has no way to know that on its own.

    One ghost point is taken from the far side at each end, ``np.gradient`` is
    applied to the extended array, and the ghosts trimmed. Going through
    ``np.gradient`` rather than differencing by hand keeps its treatment of
    non-uniform spacing, and keeps the interior identical to what it was.

    Parameters
    ----------
    f : ndarray
        the field, indexed [x,y(,z)]
    coords : 1D ndarray
        grid coordinates along this axis
    axis : int
    periodic : bool, default=False
    edge_order : int, default=1
        passed to np.gradient, which uses it only at the ends of the array. It
        therefore has no effect on a periodic axis, where the ends are ghost
        points that get trimmed -- but callers differ on it (get_vorticity takes
        numpy's default, calculate_DuDt asks for 2), so it is threaded through
        rather than fixed, and the non-periodic path stays exactly what it was.

    Returns
    -------
    ndarray, the same shape as f
    '''

    if not periodic:
        return np.gradient(f, coords, axis=axis, edge_order=edge_order)

    # The point before index 0 is index -2, not -1: the last line repeats the
    # first. Likewise the point after the last is index 1.
    period = coords[-1] - coords[0]
    x = np.concatenate(([coords[-2] - period], coords, [coords[1] + period]))
    ext = np.concatenate((np.take(f, [-2], axis=axis), f,
                          np.take(f, [1], axis=axis)), axis=axis)
    grad = np.gradient(ext, x, axis=axis, edge_order=edge_order)
    trim = [slice(None)]*f.ndim
    trim[axis] = slice(1, -1)
    return grad[tuple(trim)]



def _vorticity_from_field(flow, flow_points, periodic_dim):
    '''Curl of a single-time velocity field held as raw ndarrays.

    What ``FluidData.get_vorticity`` computes with, exposed separately for
    callers that already hold the raw arrays -- ``get_vorticity(time=)`` goes
    through the interpolant and can therefore trigger a load.

    Parameters
    ----------
    flow : sequence of ndarrays
        one velocity component per spatial dimension, each indexed [x,y(,z)]
        with **no** leading time axis
    flow_points : tuple of 1D ndarrays
        the spatial grid, one coordinate array per dimension
    periodic_dim : sequence of bool
        whether each spatial axis is periodic. A periodic axis carries a
        duplicated end line, so the field continues past either end and is
        differenced across the wrap; getting this wrong leaves the interior
        right and the outermost ring several percent off.

    Returns
    -------
    ndarray in 2D (the scalar out-of-plane component), or a tuple of three
    ndarrays in 3D (the vector curl)
    '''

    ndim = len(flow_points)

    def d(f, axis):
        return _spatial_gradient(f, flow_points[axis], axis, periodic_dim[axis])

    if ndim == 2:
        return d(flow[1][:], 0) - d(flow[0][:], 1)

    dvxdy = d(flow[0][:], 1)
    dvxdz = d(flow[0][:], 2)
    dvydx = d(flow[1][:], 0)
    dvydz = d(flow[1][:], 2)
    dvzdx = d(flow[2][:], 0)
    dvzdy = d(flow[2][:], 1)
    return (dvzdy - dvydz, dvxdz - dvzdx, dvydx - dvxdy)



def _infer_domain_edges(c):
    '''Locate a cell-centered axis' domain boundaries from its own spacing.

    Half the distance to the neighboring center: exact on a uniform grid, a
    guess on a stretched one. See center_cell_regrid for why it is a guess.
    '''
    return c[0] - (c[1] - c[0])/2, c[-1] + (c[-1] - c[-2])/2



def center_cell_regrid(flow, flow_points, periodic_dim=None, bounds=None):
    '''Extend cell-centered fluid data out to the edges of its domain.

    Finite-volume solvers specify velocity at the center of each cell, so the
    outermost samples sit half a cell inside the domain and the data reports it
    one cell narrower than it is. This adds one grid plane at each end of each
    spatial axis, at the domain boundary, and fills it by extending the field.

    The result is rectilinear but not uniform: the two outermost intervals along
    each axis are the half-cell from the outermost centers out to the boundary.
    That is the same shape ``OpenFOAMData`` builds when it splices real boundary
    patches on, and the reason to prefer patches when they exist -- this is an
    extrapolation of the interior, whereas a patch carries the boundary
    condition the solver actually applied.

    **Where the boundary is, is underdetermined.** n cell centers give n
    equations in the n+1 cell faces. Only the two outermost faces are needed
    here, and ``_infer_domain_edges`` takes them half the distance to the
    neighboring center -- exact on a uniform grid, but on a stretched one a
    biased guess: first two cells of width w and rw give w(1+r)/4 against a true
    w/2. A warning names any axis inferred that way. Pass ``bounds`` for an end
    whose true coordinate is known, as ``OpenFOAMData`` does from its patches.

    Parameters
    ----------
    flow : list of ndarray
        one array per velocity component, each indexed [x,y(,z)] for
        time-invariant flow or [t,x,y(,z)] for time-varying. Not modified.
    flow_points : tuple of ndarray
        1D cell-center coordinates along each spatial axis, increasing. At least
        two per axis, since the boundary is located from the spacing.
    periodic_dim : list of bool, optional
        per spatial axis; defaults to non-periodic throughout. A periodic axis
        wraps rather than extrapolating, so both of its new planes hold the same
        values and the field stays periodic on the returned grid.
    bounds : sequence of (float or None, float or None), optional
        the true domain boundary of each spatial axis, where it is known. A None
        entry -- or None throughout -- is inferred as above. Supplying an end
        both places the new plane exactly and extends the field to it, so the
        two need not be half a cell out.

    Returns
    -------
    flow : list of ndarray
        each grown by two along every spatial axis
    flow_points : tuple of ndarray
        cell centers with the two boundary coordinates added, in the same
        coordinate system as the input -- shifting to the first quadrant is the
        caller's business, as it is for the loaders' own grids
    '''

    ndim = len(flow_points)
    flow_points = [np.asarray(c, dtype=float) for c in flow_points]
    flow = [np.asarray(f) for f in flow]

    if periodic_dim is None:
        periodic_dim = [False]*ndim
    for d, c in enumerate(flow_points):
        if len(c) < 2:
            raise ValueError(
                "Axis {} has {} grid point(s). At least two are needed: the "
                "domain boundary is located from the spacing between "
                "them.".format('xyz'[d], len(c)))
    if flow[0].ndim == ndim:
        toff = 0
    elif flow[0].ndim == ndim + 1:
        toff = 1                          # leading time axis
    else:
        raise ValueError(
            "Flow components have {} dimensions against {} spatial axes; "
            "expected {} (time-invariant) or {} (time-varying).".format(
                flow[0].ndim, ndim, ndim, ndim+1))

    if bounds is None:
        bounds = [(None, None)]*ndim

    ##### Resolve each end, and say where that was a guess rather than exact ###
    edges = []
    inferred = []
    for d, c in enumerate(flow_points):
        guess = _infer_domain_edges(c)
        ends = tuple(guess[e] if bounds[d][e] is None else float(bounds[d][e])
                     for e in (0, 1))
        if not (ends[0] < c[0] and ends[1] > c[-1]):
            raise ValueError(
                "Domain bounds ({:g}, {:g}) for axis {} do not lie outside its "
                "cell centers ({:g}, {:g}).".format(
                    ends[0], ends[1], 'xyz'[d], c[0], c[-1]))
        edges.append(ends)
        # Only an end that was actually inferred can be wrong, and only a
        # stretched axis makes the inference more than a restatement of dx.
        if any(bounds[d][e] is None for e in (0, 1)) and \
                not np.allclose(np.diff(c), c[1]-c[0], rtol=1e-6):
            inferred.append(d)

    if len(inferred) > 0:
        warnings.warn(
            "Cell-center spacing is not uniform along {}, so the domain "
            "boundary there is inferred as half the distance to the "
            "neighboring cell center. That is exact only for a uniform grid; "
            "on a stretched one it is biased by the local stretch ratio. Pass "
            "bounds= if the true extent is known.".format(
                ', '.join('xyz'[d] for d in inferred)), UserWarning)

    ##### Extend one axis at a time #####
    # Axis by axis rather than by assembling the shell of boundary points and
    # interpolating it in one go. Sweeping this way gets the edges and corners
    # for free -- the second axis extends planes the first has already extended,
    # which is exactly the tensor-product (multilinear) extension those corners
    # need -- and the axes commute, so the order does not matter.
    new_points = []
    for d in range(ndim):
        c = flow_points[d]
        ax = d + toff
        h_lo = c[0] - edges[d][0]
        h_hi = edges[d][1] - c[-1]
        new_points.append(np.concatenate(([edges[d][0]], c, [edges[d][1]])))

        for ii, f in enumerate(flow):
            v0 = np.take(f, 0, axis=ax); v1 = np.take(f, 1, axis=ax)
            u0 = np.take(f, -1, axis=ax); u1 = np.take(f, -2, axis=ax)
            if periodic_dim[d]:
                # The two edges are the same place, so they hold the same value:
                # the field across the wrap, weighted by the two half-widths
                # meeting there.
                lo = hi = (h_lo*u0 + h_hi*v0)/(h_lo + h_hi)
            else:
                lo = v0 - (v1 - v0)*h_lo/(c[1] - c[0])
                hi = u0 + (u0 - u1)*h_hi/(c[-1] - c[-2])
            flow[ii] = np.concatenate((np.expand_dims(lo, ax), f,
                                       np.expand_dims(hi, ax)), axis=ax)

    return flow, tuple(new_points)


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
        return _linear_blend(self.flow_times, val, self.flow.__getitem__)


    def __getitem__(self, pos):
        '''
        Allows indexing into the interpolator at original time mesh points.
        '''
        return _spline_index(self, pos)

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

    def __init__(self, flow_times, flow, extrapolate=(True, True),
                 bc_type='not-a-knot'):
        '''
        Creates a PPoly instance spline instance with some additional info
        and capabilities. Will throw a custom error if times are requested
        outside of spline time bounds and extrapolate is False on that side.

        bc_type is passed through to scipy.interpolate.CubicSpline.

        Note that this class splines a whole dataset at once; dynamic loading
        uses LinearSpline, which is extensible one window at a time.
        '''
        super().__init__(flow_times, flow, axis=0, extrapolate=True,
                         bc_type=bc_type)

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



    def __call__(self, val):
        if (val < self.x[0] and not self.extrapolate[0]) \
              or (val > self.x[-1] and not self.extrapolate[1]):
            raise SplineRangeError('Out of range without extrapolation.')
        return super().__call__(val)

    def __getitem__(self, pos):
        '''
        Allows indexing into the interpolator at original time mesh points.
        '''
        return _spline_index(self, pos)

    def __setitem__(self, pos, val):
        raise RuntimeError("Cannot assign to spline object. "+
                           "Use regenerate_data to recreate original data first.")
    
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
    spatial_shape : tuple
        Shape of a single time slice of one velocity component, ``(i,j,[k])``.
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
        points at that time. Calling rather than indexing is what lets the object 
        catch times outside the currently loaded window and dynamically 
        load/spline the data needed.

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
        INUM : int > 3, True, or None (default)
            Used by subclasses to dynamically load data from storage. It
            corresponds to the number of intervals loaded at any given time when
            dynamically loading data and linearly splining. True results in
            linearly splining all data, None results in cubic splining all data.
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
        # Callbacks fired whenever fluid data lands in memory; see
        #   add_dump_observer. Set before the first _record_dump_means call
        #   below, which is the first thing that would fan out.
        self._dump_observers = []
        # Where per-dump vorticity files live, once that is known. None means
        #   nothing has established a location -- see probe_stored_vorticity.
        self.vorticity_path = None
        # The two most recently read per-dump vorticity fields, keyed on the
        #   GLOBAL dump index so the cache stays correct across a window slide
        #   it knows nothing about.
        self._vort_cache = {}

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
            self._dumps_arrived(0, flow)
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


    @property
    def spatial_shape(self):
        '''Returns the shape of one time slice of a velocity component.'''
        # fshape leads with a time axis only when the flow is time-varying, so
        # a time-invariant field is already its own spatial shape.
        if self.flow_times is None:
            return self.fshape
        return self.fshape[1:]
    


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



    @property
    def is_windowed(self):
        '''True when only a sliding window of the dataset is held in memory.

        With the whole field resident, recomputing a derived field is cheaper 
        than any I/O, so nothing is written; with a window sliding, the velocity 
        a later render would need is gone, so the derived field has to come off 
        disk.

        False for time-invariant flow, for ``INUM=None`` (cubic, all resident)
        and for ``INUM=True`` (linear, all resident) -- and also for an int
        ``INUM`` that spans the whole dataset, which holds everything and never
        slides.
        '''

        if self.flow_times is None:
            return False
        # False is normalized to None by __init__ when there are flow_times, so
        #   an int is all that is left to test.
        if self.INUM is None or self.INUM is True:
            return False
        return self.INUM < len(self.flow_times) - 1


    def add_dump_observer(self, observer):
        '''Call ``observer(idx_start, flow)`` whenever fluid data lands in memory.

        How anything deriving a per-dump quantity is told that a dump has
        arrived, without having to know which load path delivered it. Two things
        to know when writing one:

        - **The same dump can be reported more than once.** A window sliding
          back to the start of a series reloads the opening dumps, which have
          usually been reported already. So an observer that acts on each report
          -- writing a file, appending to a list -- has to remember which dumps
          it has handled and ignore a repeat.
        - **Time-invariant flow never fires it**, because no dumps arrive over
          time: the whole field is present from construction. An observer that
          needs to see every dump has to handle that case itself, at the point
          where it registers.

        Parameters
        ----------
        observer : callable
            called with ``(idx_start, flow)``, where ``idx_start`` is the index
            into ``flow_times`` of the first time point supplied and ``flow`` is
            a list of per-component ndarrays with a leading time axis -- raw
            data, not splines.
        '''

        if observer not in self._dump_observers:
            self._dump_observers.append(observer)


    def remove_dump_observer(self, observer):
        '''Stop calling an observer registered by :meth:`add_dump_observer`.
        A no-op if it is not registered.'''

        if observer in self._dump_observers:
            self._dump_observers.remove(observer)


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



    @property
    def dump_means(self):
        """Spatial mean of each velocity component, per dump, as (n_dumps, ncomp).

        NaN for a dump that has never been in memory, which is possible only
        while a window is being slid. Time-invariant flow has a single row.
        """

        return np.atleast_2d(self._dump_means)


    def iter_resident_dumps(self):
        """Yield ``(t_idx, [single-time component arrays])`` for what is in memory.

        One dump at a time, and never a stack of them: a caller that reduces
        each dump as it arrives would otherwise pay a second full copy of the
        dataset, which under ``INUM=None`` is the whole thing. Covers the
        time-invariant, all-resident and windowed cases identically, so a
        consumer needs no branch of its own.
        """

        if self.flow_times is None:
            yield 0, list(self._flow)
            return
        if self.INUM is None:
            # Cubic, everything resident. Indexing a component's spline by time
            #   index reconstructs one dump; regenerate_data would rebuild the
            #   entire series at once.
            for i in range(len(self.flow_times)):
                yield i, [f[i] for f in self._flow]
            return
        # Linear. regenerate_data hands back the resident window by reference,
        #   so slicing a dump out of it copies nothing.
        idx0 = getattr(self, 'loaded_idx_bnds', (0, None))[0]
        window = [f.regenerate_data() for f in self._flow]
        for i in range(len(window[0])):
            yield idx0 + i, [w[i] for w in window]


    def _dumps_arrived(self, idx_start, flow):
        '''Fluid data just landed in memory: cache what is cheap, tell observers.

        Called from every point where that happens, and nowhere else, so an
        observer inherits the correctness of the call sites -- including the
        forward slide's deliberate skip of the two holdover dumps it carried
        over from the outgoing window.

        Parameters
        ----------
        idx_start : int
            index into flow_times of the first time point supplied
        flow : list of ndarrays
            per-component data with a leading time axis
        '''

        self._record_dump_means(idx_start, flow)
        for observer in self._dump_observers:
            observer(idx_start, flow)


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
        means = _linear_blend(self.flow_times, time, self._dump_means.__getitem__)

        if np.isnan(means).any():
            return None
        return tuple(float(m) for m in means)



    def get_mean_velocity(self, time=None, t_idx=None):
        '''Spatial mean of each fluid velocity component.

        Served from a per-dump cache of means built as data loads, so this does
        not touch the velocity field and does not trigger a load for any time
        whose bracketing dumps have already been seen. The value is exact rather
        than approximate: both spline classes evaluate as a weighted sum of the
        nodal fields, and the spatial mean is linear, so the mean of the splined
        field is the splined mean.

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
        '''Subclasses should implement this method to load additional data.

        Return one array per velocity component. A leading time axis is optional:
        :meth:`_load_dumps`, which is what the slider calls, adds one where a
        single-dump read left it off.
        '''
        raise NotImplementedError('The subclass for this type of data must '+
                                  'implement its own data loaders.')


    def _load_dumps(self, d_start, d_finish):
        """load_dumpfiles, with the leading time axis guaranteed.

        ``update_spline`` concatenates what a load returns against the resident
        window and ``_record_dump_means`` reduces over its leading axis, so the
        axis has to be there even for a single dump. Some of the readers behind
        ``load_dumpfiles`` drop it in that case -- correctly, for the
        constructor's one-shot read of a time-invariant field, and wrongly here.
        Every load on the streaming path comes through this method, so a subclass
        implementing ``load_dumpfiles`` does not have to think about it.
        """

        ndim = len(self.flow_points)
        return [f if f.ndim > ndim else f[np.newaxis, ...]
                for f in self.load_dumpfiles(d_start, d_finish)]



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
            flow = self._load_dumps(d_start, d_finish)

            # Record means for the freshly loaded dumps, which start two time
            # points into the new window. The two holdovers prepended below are
            # already in the sidecar from when they were first loaded, and those
            # entries came from raw data rather than from a spline evaluation
            # carried across a window boundary.
            self._dumps_arrived(idx_start+2, flow)

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
                self._flow = self._load_dumps(self.d_start, self.d_start + self.INUM)
                self.loaded_dump_bnds = (self.d_start, self.d_start + self.INUM)
                self.loaded_idx_bnds = (0, self.INUM)
                self._dumps_arrived(0, self._flow)
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
                flow = self._load_dumps(d_start, d_finish)

                # Record means for the freshly loaded dumps. Sliding backward,
                # these occupy the front of the new window; the two holdovers
                # appended below already have their means recorded.
                self._dumps_arrived(idx_start, flow)

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
           **Temporarily unavailable.** Tiling will return as a
           position-wrapping implementation that works in both 2D and 3D without
           materializing the tiled field. The previous body is preserved
           commented-out below.

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
            'for 2D and 3D. See docs/notes/run_persistence.md.')

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

        # Periodic axes are differenced across the wrap; see
        # _vorticity_from_field, which is shared with the dump-arrival observer
        # so that the two cannot compute different fields.
        return _vorticity_from_field(flow, self.flow_points, self.periodic_dim)
    


    #####################   per-dump vorticity   #####################
    #
    # Vorticity is not cached -- it is sourced. Which of three things happens is
    # decided by is_windowed and by whether the source already ships the field:
    #
    #   whole field resident     -> nothing on disk; recompute from the velocity
    #   windowed, source has     -> read the source's own per-dump field
    #   windowed, source has not -> write one per dump as it lands, in the
    #                               source's own format, and read it back
    #                               through the same path
    #
    # A subclass that ships the field overrides probe_stored_vorticity and
    # read_dump_vorticity; one whose format differs from the generic rectilinear
    # default also overrides write_dump_vorticity. The blended read
    # (get_stored_vorticity) is generic.
    #
    # 2D only, since a fluid backdrop is only drawn in 2D. OpenFOAMData ships a
    # `vorticity` cell array on every dump and would be the first beneficiary if
    # a 3D backdrop arrives with the vtk plotting rewrite; it is deliberately not
    # read until then, since reading it means the cell-order permutation and
    # boundary splice again for a field nothing would draw.

    VORTICITY_TITLE = 'Omega'


    def dump_number(self, t_idx):
        """The source's own dump number for index ``t_idx`` into ``flow_times``.

        A time resolves to dumps as ``d_start + i``, uniformly: IB2d's
        ``d_start`` is the first dump number on disk, and a source with no dump
        numbering of its own is indexed from zero.
        """

        return int(getattr(self, 'd_start', 0)) + int(t_idx)


    def source_dir(self):
        """The directory this fluid was read from, or None if it came from arrays.

        Where a derived field is written by preference, so that a later run,
        ParaView, or the solver's own tooling finds it beside the velocity dumps
        with no knowledge of Planktos.
        """

        path = getattr(self, 'path', None)
        return None if path is None else Path(path)


    def probe_stored_vorticity(self):
        """Does this source already carry a per-dump vorticity field?

        ``'partial'`` is reported as such rather than rounded to either of the
        others, since a caller must not serve one dump's field for another's.

        Returns
        -------
        state : {'complete', 'partial', 'absent'}
        directory : Path or None
            where the field is, when the state is not 'absent'
        """

        return 'absent', None


    def vorticity_filename(self, t_idx):
        """Name of the per-dump vorticity file for ``t_idx``.

        The reader, the writer and anything moving the file into place all take
        the name from here, so a subclass that changes its solver's convention
        changes it once.
        """

        return '{}.{:04d}.vtk'.format(self.VORTICITY_TITLE,
                                      self.dump_number(t_idx))


    def read_dump_vorticity(self, t_idx):
        """Read one dump's vorticity field off disk, on the velocity's grid.

        Reads what :meth:`write_dump_vorticity` writes: a rectilinear-grid scalar
        vtk on the same grid as ``flow_points``. A subclass whose solver uses
        another format overrides both halves together.

        Parameters
        ----------
        t_idx : int
            index into ``flow_times``

        Returns
        -------
        ndarray indexed [x,y], on ``flow_points``
        """

        vort, _, _ = _dataio.read_vtk_Rectilinear_Grid_Scalars(
            self._vorticity_dir() / self.vorticity_filename(t_idx))
        return vort


    def write_dump_vorticity(self, t_idx, vort, path):
        """Write one dump's vorticity field, in this source's own format.

        Writes a rectilinear-grid scalar vtk, which can express any grid
        ``FluidData`` supports. A subclass on a uniform grid overrides this to
        write structured points instead, matching what its solver prints.

        Parameters
        ----------
        t_idx : int
            index into ``flow_times``
        vort : ndarray
            the field, indexed [x,y], on ``flow_points``
        path : str or Path
            directory to write into
        """

        _dataio.write_vtk_2D_rectilinear_grid_scalars(
            path, self.VORTICITY_TITLE, vort, self.flow_points,
            cycle=self.dump_number(t_idx),
            time=None if self.flow_times is None else float(self.flow_times[t_idx]),
            sep='.')


    def _vorticity_dir(self):
        """Where per-dump vorticity lives, raising if nothing has said."""

        if self.vorticity_path is None:
            raise RuntimeError(
                'no per-dump vorticity location is known for this fluid. One is '
                'established when recording starts (Environment.record), or by '
                'a source that ships the field.')
        return Path(self.vorticity_path)


    def get_stored_vorticity(self, time):
        """Vorticity at ``time``, blended from per-dump files on disk.

        The dynamic-loading counterpart of :meth:`get_vorticity`: the velocity
        that time falls between is no longer resident, so the curl is assembled
        from per-dump fields instead. The result is exactly the curl of the
        velocity in use -- ``LinearSpline`` evaluates as a weighted sum of two
        adjacent nodal fields and the curl is linear, so blending per-dump
        vorticity with the same two weights gives the curl of the blend.

        **Linear splining only**, and it raises otherwise. Cubic weights come
        from a global solve, so reconstructing from per-dump files would mean
        holding the whole series; that regime has the field resident and should
        call :meth:`get_vorticity` instead.

        Reads are served from a two-slot cache, which is one read per dump
        interval for a monotone sweep in either direction.

        Parameters
        ----------
        time : float

        Returns
        -------
        ndarray indexed [x,y]
        """

        if self.flow_times is None:
            return self._dump_vorticity(0)

        if self.INUM is None:
            raise RuntimeError(
                'this fluid is splined cubically in time, so per-dump vorticity '
                'cannot reproduce the field in use: not-a-knot weights are '
                'global, and applying them would mean holding the whole series. '
                'The whole field is resident in this regime -- call '
                'get_vorticity(time=) instead, which differentiates the '
                'interpolated velocity and is the same field by construction.')

        # Through the same weights the velocity itself is blended with, so the
        #   result is the curl of the field in use rather than an approximation
        #   of it. Reads only the dumps the interpolation needs.
        return _linear_blend(self.flow_times, time, self._dump_vorticity)


    def _dump_vorticity(self, t_idx):
        """One dump's vorticity, from the two-slot read cache."""

        t_idx = int(t_idx)
        if t_idx not in self._vort_cache:
            self._vort_cache[t_idx] = self.read_dump_vorticity(t_idx)
        vort = self._vort_cache[t_idx]
        # Evict whatever is furthest from what was just asked for. On a monotone
        #   sweep that is the trailing dump, which is what makes one read per
        #   dump interval enough in either direction.
        while len(self._vort_cache) > 2:
            drop = max(self._vort_cache, key=lambda k: abs(k - t_idx))
            del self._vort_cache[drop]
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

        # Per axis rather than one call over all of them, because periodicity is
        # a per-axis property: a periodic axis is differenced across the wrap,
        # where np.gradient would one-side against data that actually continues.
        # np.array(flow) puts the components on axis 0, so spatial axis d is
        # array axis d+1.
        stacked = np.array(flow)
        flow_grad = [_spatial_gradient(stacked, self.flow_points[d], d+1,
                                       self.periodic_dim[d], edge_order=2)
                     for d in range(self.ndim)]

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

        If INUM (interval number) is set to an integer >= 4, then the data 
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
        INUM : int > 3, True, or None (default)
            max number of splined intervals held at any one time; the number of
            time points held is 1+INUM, and INUM must be at least 4. None splines
            the entire dataset at once and cubically in time; True holds the
            entire dataset too but splines it linearly; an int streams a sliding
            window from storage and splines that linearly.
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



    def probe_stored_vorticity(self):
        """Look for IB2d's own Omega.####.vtk beside the velocity dumps.

        IB2d writes vorticity only if the run's ``input2d`` asked for it, so it
        is genuinely optional.
        """

        # Checked against the whole dump range rather than by existence of any
        #   file, so that a series covering part of the run reads as partial.
        directory = Path(self.path)
        present = set()
        for f in directory.glob('{}.*.vtk'.format(self.VORTICITY_TITLE)):
            stem = f.name[len(self.VORTICITY_TITLE)+1:-4]
            try:
                present.add(int(stem))
            except ValueError:
                continue
        needed = set(range(self.d_start, self.d_finish+1))
        if not present & needed:
            return 'absent', None
        if needed <= present:
            return 'complete', directory
        return 'partial', directory


    def read_dump_vorticity(self, t_idx):
        """Read one dump's Omega.####.vtk, on the same grid as the velocity.

        Goes through the same reader the velocity does, and needs the same two
        adjustments after it: the read returns ``[y,x]``, and IB2d omits the
        periodic endpoint in each direction, so a 6x5 dump becomes a 7x6 field
        matching ``flow_points``.
        """

        # No LLC shift here -- flow_points is already in quadrant 1, and the
        #   field carries no coordinates of its own.
        vort = _dataio.read_2DEulerian_Data_From_vtk(
            self._vorticity_dir(), '{:04d}'.format(self.dump_number(t_idx)),
            self.VORTICITY_TITLE)
        return _wrap_scalar(vort.T, self.periodic_dim)


    def write_dump_vorticity(self, t_idx, vort, path):
        """Write one dump's vorticity in the form IB2d prints its own.

        Structured points rather than the generic rectilinear form: the IB2d
        Eulerian grid is always regular, and structured points is what the solver
        writes. The wrap is stripped and the field transposed, both inverses of
        what :meth:`read_dump_vorticity` does.
        """

        stripped = _unwrap_scalar(vort, self.periodic_dim)
        # The source's own coordinates, not the quadrant-1 shift Planktos works
        #   in, so the file sits on exactly the grid the solver's own dumps do.
        grid = tuple(self._orig_flow_points[d] + self.fluid_domain_LLC[d]
                     for d in range(2))
        _dataio.write_vtk_structured_points_scalars(
            path, self.VORTICITY_TITLE, stripped, grid,
            cycle=self.dump_number(t_idx),
            time=None if self.flow_times is None else float(self.flow_times[t_idx]),
            sep='.')


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

        If INUM (interval number) is set to an integer >= 4, then the data 
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
        INUM : int > 3, True, or None (default)
            max number of splined intervals held at any one time; the number of
            time points held is 1+INUM, and INUM must be at least 4. None splines
            the entire dataset at once and cubically in time; True holds the
            entire dataset too but splines it linearly; an int streams a sliding
            window from storage and splines that linearly.
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
    
        # A plane exported as 3D vtk is 2D data. Drop any axis one point
        #   thick, with its velocity component; the axes are remembered so that
        #   later windowed reads are collapsed the same way.
        self._read_ndim = len(mesh)
        flow, mesh, periodic_dim, self._flat = _collapse_flat_axes(
            flow, mesh, periodic_dim, source=str(path))

        # shift domain to quadrant 1
        flow_points = tuple(m - m[0] for m in mesh)
        fluid_domain_LLC = tuple(m[0] for m in mesh)
        # It is assumed that the fluid spatial grid includes all 
        # domain boundaries.
        self.L = [fp[-1] for fp in flow_points]
        
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
        return _drop_flat_axes(flow, self._flat, self._read_ndim)



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

        # Only the time axis is dropped, and only for a single dump. A blanket
        #   squeeze would also collapse a spatial axis one point thick -- which
        #   is a plane, and is _collapse_flat_axes' business, since the
        #   coordinate arrays have to go with it.
        flow = [np.array(f) for f in flow]
        if d_start == d_finish:
            flow = [f[0] for f in flow]

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

        # _read_vtufile hands back a two-entry mesh for 2D data, so the grid is
        #   built over however many axes there are rather than always three.
        flow, mesh, periodic_dim, _ = _collapse_flat_axes(
            flow, mesh, periodic_dim, source=str(path))

        # shift domain to quadrant 1
        flow_points = tuple(m - m[0] for m in mesh)
        fluid_domain_LLC = tuple(m[0] for m in mesh)
        # It is assumed that the fluid spatial grid includes all 
        # domain boundaries.
        self.L = [fp[-1] for fp in flow_points]

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

        # One component per axis of the mesh: the 2D branch above never fills
        #   flow[2], so carrying it through would hand back an empty array as a
        #   third velocity component.
        flow = [np.array(f) for f in flow[:len(mesh)]]
        if len(flow_times) == 1:
            flow = [f[0] for f in flow]

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

        Because the mesh is read once and every later dump is reshaped through
        it, the second dump is checked against it automatically, and a changed
        cell count is caught on every dump. See ``_verify_dump_mesh``.

        Despite the ``vtkUnstructuredGrid`` container, the mesh is required to be
        rectilinear -- Planktos interpolates on a tensor-product grid. That is
        verified rather than assumed; see ``_build_lattice``.

        A dump that the series index declares but which is not on disk is skipped
        with a warning, and the timeline is built densely over the dumps that do
        exist. A truncated or interrupted export is an ordinary thing to be
        handed, and discovering the gap partway through a long streaming run
        would be the worst possible moment for it.

        **Pieces of the export may also be missing entirely**, and the timeline
        is recovered from whatever remains, in this order:

        1. the ``.vtm.series`` index, using the times it declares;
        2. failing that, the ``.vtm`` manifests, using the ``TimeValue`` each
           one carries;
        3. failing that, the dump directories, using the ``TimeValue`` in each
           ``internal.vtu``;
        4. failing that, unit time steps.

        Every step past the first warns, and the one taken is recorded in
        ``dump_source`` and ``time_source``: a run completing on a timeline other
        than the one the user believes they loaded is the failure this most needs
        to avoid. Dumps discovered by globbing (2 and 3) are ordered by their
        recovered times, falling back to a numeric-aware sort of their names.

        If INUM (interval number) is set to an integer >= 4, the data will be
        dynamically loaded as needed with INUM intervals between the temporal
        data sets available at any given time.

        Parameters
        ----------
        path : string
            path to the directory holding the export (typically the ``VTK``
            directory foamToVTK writes), or to the ``.vtm.series`` index itself.
            A directory need not contain an index; see the fallback chain above.
        INUM : int > 3, True, or None (default)
            max number of splined intervals held at any one time; the number of
            time points held is 1+INUM, and INUM must be at least 4. None splines
            the entire dataset at once and cubically in time; True holds the
            entire dataset too but splines it linearly; an int streams a sliding
            window from storage and splines that linearly.
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
            default. False fills any uncovered face by extrapolating the
            interior out to it instead, with a warning -- per face, so every
            face that does have a patch still uses it. Prefer a patch where one
            exists: it carries the boundary condition the solver applied, and a
            no-slip wall is not the linear extension of the flow beside it.

        Attributes
        ----------
        dump_source : {'series', 'manifests', 'directories'}
            which step of the fallback chain supplied the list of dumps
        time_source : string
            a phrase naming where the times came from, for the record and for
            error messages. Not drawn from a fixed set -- it says how many gaps
            a secondary source filled, where that applies.
        series_path : Path or None
            the ``.vtm.series`` index, if there was one to read
        '''

        ##### Parse parameters #####
        self.path = path
        self.vel_conv = vel_conv
        self.require_boundary = require_boundary
        # FluidData.__init__ runs last and is what sets self.periodic_dim, but
        # the regrid of an uncovered face needs it during the opening load.
        self._periodic_dim = periodic_dim
        # The boundary-condition corner warning is a property of the dataset,
        # not of the timestep, so say it once rather than on every window slide.
        self._warned_bc_corner = False
        # The mesh check runs on the second dump and only the first time it is
        # read -- a window sliding back to the start must not pay for it again.
        self._mesh_verified = False
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
            # Dump 1 is always in the opening load -- the whole series if it
            # fits, else dumps 0..INUM -- so the mesh check lands at
            # construction and costs no read of its own.
            verify = n == 1 and not self._mesh_verified
            vel = self._read_dump(self._dumps[n], verify=verify)
            if verify:
                self._mesh_verified = True
            for dim in range(3):
                flow[dim].append(vel[..., dim])
        flow = [np.array(f) for f in flow]

        if self.vel_conv is not None:
            for ii, d in enumerate(flow):
                flow[ii] = d*self.vel_conv
        return flow



    @staticmethod
    def _natural_key(name):
        '''Sort key that orders the numbers embedded in a name numerically.

        foamToVTK numbers its dumps without zero padding -- case08_..._787
        through case08_..._1034 -- so a lexical sort puts 1008 ahead of 787 and
        silently reverses part of the timeline. The split always alternates
        non-digit, digit, ..., so two keys compare like with like throughout.
        '''
        return tuple(int(s) if s.isdigit() else s
                     for s in re.split(r'(\d+)', name))



    @staticmethod
    def _check_internal(datasets, source):
        if 'internal' not in datasets:
            raise RuntimeError(
                "Manifest {} names no 'internal' dataset. Expected the "
                "volume data written by foamToVTK.".format(source.name))



    def _find_dumps(self, path):
        '''Locate the dumps of the series, and how each one is timestamped.

        Three sources are tried in turn, each a fallback for the one before it:
        the ``.vtm.series`` index, the ``.vtm`` manifests, and the dump
        directories themselves. Which one answered is recorded in
        ``dump_source`` / ``time_source``, and warned about whenever it is not
        the first: a degraded timeline must never be accepted in silence.

        Returns
        -------
        list of (datasets, time), one per dump the source declares, in the order
            the source gives them. datasets is the resolved {dataset name: path}
            mapping, or None for a dump that is declared but whose data is not
            on disk; time is NaN for a dump the source could not timestamp.
        '''

        path = Path(path)
        if path.is_file():
            return self._candidates_from_series(path)
        if not path.is_dir():
            raise FileNotFoundError("{} not found!".format(str(path)))

        ##### 1: the .vtm.series index #####
        found = sorted(path.glob('*.vtm.series'))
        if len(found) > 1:
            raise RuntimeError(
                "Found {} .vtm.series indices in {}: {}. Pass the one to "
                "read.".format(len(found), str(path),
                               [f.name for f in found]))
        if len(found) == 1:
            return self._candidates_from_series(found[0])

        ##### 2: the .vtm manifests #####
        vtms = sorted(path.glob('*.vtm'),
                      key=lambda p: self._natural_key(p.name))
        if len(vtms) > 0:
            warnings.warn(
                "No .vtm.series index in {}; falling back to the {} .vtm "
                "manifest(s) found there, timed by their own TimeValue "
                "entries.".format(str(path), len(vtms)), UserWarning)
            return self._candidates_from_manifests(vtms)

        ##### 3: the dump directories #####
        dumpdirs = sorted((d for d in path.iterdir()
                           if d.is_dir() and (d/'internal.vtu').is_file()),
                          key=lambda p: self._natural_key(p.name))
        if len(dumpdirs) > 0:
            warnings.warn(
                "No .vtm.series index and no .vtm manifests in {}; falling "
                "back to the {} dump director(ies) found there, timed by the "
                "TimeValue in each internal.vtu. Boundary patches are taken to "
                "be the .vtp files beside each internal.vtu or in a boundary/ "
                "subdirectory.".format(str(path), len(dumpdirs)), UserWarning)
            return self._candidates_from_dirs(dumpdirs)

        raise FileNotFoundError(
            "No .vtm.series index, .vtm manifest, or dump directory holding an "
            "internal.vtu found in {}. Expected the VTK directory written by "
            "foamToVTK.".format(str(path)))



    def _candidates_from_series(self, series):
        '''Dumps declared by a .vtm.series index -- the primary source.'''

        self.series_path = series
        self.dump_source = 'series'
        self._source_label = "the index {}".format(series.name)
        self.time_source = "the .vtm.series index"

        files, times = _dataio.read_vtm_series(series)

        candidates = []; filled = 0
        for f, t in zip(files, times):
            if not f.is_file():
                candidates.append((None, t))
                continue
            datasets, mtime = _dataio.read_vtm_manifest(f)
            self._check_internal(datasets, f)
            if np.isnan(t):
                # The index failed to describe this entry; the manifest's own
                # TimeValue is the per-file fallback for precisely that.
                t = np.nan if mtime is None else mtime
                filled += int(not np.isnan(t))
            if not datasets['internal'].is_file():
                candidates.append((None, t))
                continue
            candidates.append((datasets, t))

        if filled > 0:
            self.time_source = ("the .vtm.series index, with per-manifest "
                                "TimeValue filling {} gap(s) in it".format(filled))
            warnings.warn(
                "{} declares {} dump(s) without a usable time; their TimeValue "
                "was taken from the .vtm manifest instead.".format(
                    series.name, filled), UserWarning)

        return candidates



    def _candidates_from_manifests(self, vtms):
        '''Dumps found by globbing .vtm manifests, timed by their TimeValue.

        The fallback for an export whose .vtm.series index was never written or
        did not survive the transfer. Everything the index would have supplied
        -- which dumps exist, and when each one is -- is present in the
        manifests themselves, one small XML parse apiece.
        '''

        self.series_path = None
        self.dump_source = 'manifests'
        self._source_label = "the .vtm manifests in {}".format(
            str(vtms[0].parent))
        self.time_source = "the TimeValue of each .vtm manifest"

        candidates = []
        for f in vtms:
            datasets, mtime = _dataio.read_vtm_manifest(f)
            self._check_internal(datasets, f)
            t = np.nan if mtime is None else mtime
            candidates.append(
                (datasets if datasets['internal'].is_file() else None, t))
        return candidates



    def _candidates_from_dirs(self, dumpdirs):
        '''Dumps found by globbing the dump directories themselves.

        The last resort: no index and no manifests, so both the dump list and
        the timeline have to come out of the data files. Time is read from each
        internal.vtu's TimeValue header rather than by parsing the file, since
        an unstructured export repeats its whole mesh every dump and parsing all
        of them to recover one float apiece would read gigabytes.
        '''

        self.series_path = None
        self.dump_source = 'directories'
        self._source_label = "the dump directories in {}".format(
            str(dumpdirs[0].parent))
        self.time_source = "the TimeValue in each internal.vtu"

        candidates = []; untimed_writer = False
        for k, d in enumerate(dumpdirs):
            internal = d/'internal.vtu'
            t = _dataio.read_vtkxml_time_only(internal)
            if t is None and not untimed_writer:
                # The header scan is bounded and decodes only the common
                # encodings, so a miss is not proof the file is untimed. Pay for
                # one full read before concluding it.
                t = _dataio.read_vtkxml_cell_data(
                    internal, arrays=(), load_cell_coordinates=False)[2]
                if t is None and k == 0:
                    # Whether a writer records TimeValue at all is a property of
                    # the export, not of one file within it. Settling that on the
                    # first dump keeps "this series carries no times" from
                    # costing a full parse of every dump in it -- gigabytes of
                    # static mesh re-read, for a conclusion the first file has
                    # already given.
                    untimed_writer = True
            # foamToVTK puts the patches in a boundary/ subdirectory; tolerate
            # them sitting beside internal.vtu as well.
            datasets = {'internal': internal}
            for p in sorted((d/'boundary').glob('*.vtp')) + sorted(d.glob('*.vtp')):
                datasets.setdefault(p.stem, p)
            candidates.append((datasets, np.nan if t is None else t))
        return candidates



    def _read_series(self, path):
        '''Resolve the dump series and work out which of its dumps are present.

        Returns
        -------
        dumps : list of dict
            the resolved {dataset name: path} manifest of each dump that exists,
            in time order
        flow_times : ndarray of floats, or None if there is only one dump
            time of each entry of dumps, shifted so the first is 0.0
        '''

        # Eagerly, at construction, rather than at the window slide that needs a
        # missing file: under dynamic loading that raise lands arbitrarily deep
        # into a run, which is the worst possible moment and exactly what
        # streaming makes likely.
        candidates = self._find_dumps(path)

        ##### Drop the dumps whose data is not on disk #####
        def _fmt(t):
            return 'unknown' if t is None or np.isnan(t) else '{:g}'.format(t)

        dumps = []; keep_times = []; missing = []
        for datasets, t in candidates:
            if datasets is None:
                missing.append(_fmt(t))
            else:
                dumps.append(datasets)
                keep_times.append(t)

        if len(dumps) == 0:
            raise FileNotFoundError(
                "None of the {} dumps found via {} are present on disk.".format(
                    len(candidates), self._source_label))

        if len(missing) > 0:
            # Warn rather than fail: a truncated or interrupted export is normal.
            # But warn loudly -- silence here would be worse than the failure,
            # since the run would complete with nothing indicating that the
            # timeline is not the one the source declared.
            warnings.warn(
                "{} of {} dumps found via {} are not on disk and have been "
                "skipped: t = {}. The timeline is built over the {} dumps that "
                "remain.".format(len(missing), len(candidates),
                                 self._source_label, ', '.join(missing),
                                 len(dumps)), UserWarning)

        if len(dumps) == 1:
            return dumps, None

        keep_times = np.array(keep_times, dtype=float)

        ##### No time information anywhere: fall back to unit steps #####
        if np.all(np.isnan(keep_times)):
            self.time_source = 'assumed unit steps'
            warnings.warn(
                "No time information for any dump found via {}: assuming unit "
                "time steps. Every time the simulation is run against is "
                "therefore an index, not a physical time, and any velocity in "
                "physical units is scaled wrongly by the true dump "
                "interval.".format(self._source_label), UserWarning)
            return dumps, np.arange(len(dumps), dtype=float)

        if np.any(np.isnan(keep_times)):
            # Deliberately not the unit-step fallback that VTK3dData takes when
            # any single dump is untimed. Unit steps are defensible only when
            # nothing better exists; here something does, for most of the
            # series, and overwriting a real timeline with indices would move
            # every dump that *was* timed to the wrong place.
            raise RuntimeError(
                "No time information for {} of {} dumps found via {}, but the "
                "rest are timed. Unit time steps would misplace the dumps that "
                "do carry a time, so the series cannot be read as it "
                "stands.".format(int(np.isnan(keep_times).sum()), len(dumps),
                                 self._source_label))

        ##### Order the dumps we globbed ourselves by their recovered times ####
        # A .vtm.series declares its own order and is authoritative about it, so
        # it is left as written. For a globbed source the order was ours to pick
        # and the filenames were only a proxy; the times are the real thing.
        if self.dump_source != 'series':
            order = np.argsort(keep_times, kind='stable')
            if not np.array_equal(order, np.arange(len(order))):
                warnings.warn(
                    "The dumps found via {} are not in time order under their "
                    "filenames; they have been reordered by their recorded "
                    "times.".format(self._source_label), UserWarning)
                dumps = [dumps[i] for i in order]
                keep_times = keep_times[order]

        # shift so that the first dump loaded corresponds to environment time 0
        flow_times = keep_times - keep_times[0]

        ##### The timeline has to be a timeline #####
        # Both splines divide by the interval between successive times, so a
        # repeat or a step backward is not a degraded timeline but an unusable
        # one. Nothing above can produce it from well-formed input, which is
        # exactly why it is worth saying out loud if it appears.
        bad = np.nonzero(np.diff(flow_times) <= 0)[0]
        if len(bad) > 0:
            raise RuntimeError(
                "Dump times from {} are not strictly increasing: t = {} is "
                "followed by t = {}.".format(
                    self.time_source, '{:g}'.format(keep_times[bad[0]]),
                    '{:g}'.format(keep_times[bad[0]+1])))

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
        and splitting at gaps supplies it.

        rel_tol is a fraction of the axis' full span, and is not delicate: it
        needs only to sit between the roundoff spread and the true grid spacing.
        Any value from 1e-3 to 1e-8 gives identical results on real export data.
        '''

        # np.unique will not do here: cells that share a level do not always land
        #   on the same float64. On a 775k-cell export a level held up to 8
        #   distinct values spanning 5e-16 of a cell width, and np.unique
        #   reported 79 levels where 66 existed.
        order = np.argsort(v, kind='stable')
        s = v[order]
        tol = (s[-1] - s[0])*rel_tol
        brk = np.nonzero(np.diff(s) > tol)[0]
        starts = np.concatenate(([0], brk+1))
        ends = np.concatenate((brk+1, [len(s)]))

        # Any member of a level would serve as its coordinate -- the spread is
        #   pure roundoff -- but the mean is the obvious choice.
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



    def _verify_dump_mesh(self, centers, perm, shape, axes, source,
                          plane=None, rel_tol=1e-5):
        '''Check that a dump's cells still lie where the mesh says they do.

        The mesh is read once and every later dump reshaped through its
        permutation, which is sound for one OpenFOAM run and not for a series
        stitched from two. A changed cell count is caught elsewhere, on every
        dump; a reordering at the same count is the dangerous one, since the
        reshape succeeds and every value lands in the wrong place.

        Only the second dump is checked, since cell ordering is a property of the
        writer. `_read_dump` takes the flag per call, so widening it to every
        dump is a one-line change at the caller.

        Parameters
        ----------
        centers : Nx3 ndarray
            cell centers of this dump, in the file's own order. For a patch file
            holding several faces this is all of them; perm selects.
        perm : ndarray of int
            the selection/reordering established at construction. Its length,
            not that of centers, is what must match shape.
        shape : tuple of int
            lattice shape the permuted cells are expected to fill
        axes : tuple of int
            spatial axes the block varies over, in the order of shape
        source : string
            name of the file, for the error message
        plane : (int, float), optional
            axis and coordinate of the constant direction of a planar patch
        rel_tol : float, default=1e-5
            tolerance as a fraction of each axis' span. The same figure
            _cluster_axis separates levels with: a smaller deviation could not
            have moved a cell to another level, so it cannot change where a
            value lands.
        '''

        ordered = centers[perm].reshape(*shape, centers.shape[1])
        # _grid[d] is [boundary plane, *interior cell centers, boundary plane],
        # so the interior lattice -- which is what the permutation indexes, and
        # what a patch shares in its two tangential directions -- is the middle.
        # NB this assumes the boundary splice happened. _build_grid always puts
        # an edge coordinate at each end, inferred or from a patch, so the slice
        # still means "the cell centers" under require_boundary=False.
        def along(k):
            '''grid vector of the k'th axis of shape, shaped to broadcast'''
            return self._grid[axes[k]][1:-1].reshape(
                [-1 if j == k else 1 for j in range(len(shape))])

        # A planar patch's constant coordinate is a scalar and broadcasts as is.
        checks = [(d, along(k)) for k, d in enumerate(axes)]
        if plane is not None:
            checks.append(plane)

        for d, ref in checks:
            off = float(np.abs(ordered[..., d] - ref).max())
            tol = float(np.ptp(self._grid[d]))*rel_tol
            if off > tol:
                raise RuntimeError(
                    "The cells of {} do not lie on the mesh established from "
                    "the first dump: along {}, they are off by up to {:g} "
                    "(tolerance {:g}). The series is assumed to be one run, "
                    "with one fixed cell ordering; a dump whose cells are "
                    "merely reordered would otherwise load with every value in "
                    "the wrong place.".format(source, 'xyz'[d], off, tol))



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
        self._patch_cells = {}
        for name, fname in datasets.items():
            if name == 'internal':
                continue
            pcenters, _, _ = _dataio.read_vtkxml_cell_data(fname, arrays=())
            # Every later dump indexes this patch with the selection built here,
            # so a patch that changes length would take the wrong cells (or
            # raise a bare IndexError). Cheap to check, so it always is.
            self._patch_cells[name] = len(pcenters)
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

        # Faces with no patch. Kept per face rather than per dataset: a patch
        # carries the boundary condition the solver applied, which extrapolating
        # the interior cannot, so every face that has one keeps using it and
        # only the uncovered faces are regridded.
        self._regrid_faces = [(d, side) for d in range(3) for side in (0, 1)
                              if (d, side) not in self._faces]
        if len(self._regrid_faces) > 0:
            names = ', '.join('{}{}'.format('xyz'[d], '-+'[s])
                              for d, s in self._regrid_faces)
            if self.require_boundary:
                raise RuntimeError(
                    "No boundary patch covers the {} face(s) of the "
                    "domain. Cell centers are inset half a cell, so without "
                    "them the domain would be reported short by half a cell "
                    "there, shifting every coordinate in it with nothing "
                    "downstream able to detect it. Pass require_boundary=False "
                    "to extrapolate the interior out to those faces "
                    "instead.".format(names))
            stretched = [d for d in range(3)
                         if not np.allclose(np.diff(interior[d]),
                                            interior[d][1]-interior[d][0],
                                            rtol=1e-6)
                         and any((d, s) in self._regrid_faces for s in (0, 1))]
            warnings.warn(
                "No boundary patch covers the {} face(s); the fluid there is "
                "extrapolated from the interior and the domain edge is placed "
                "half a cell out. A patch would carry the boundary condition "
                "the solver applied, which this cannot -- a no-slip wall, for "
                "one, is not the linear extension of the flow beside it.".format(
                    names) +
                ("" if len(stretched) == 0 else
                 " Cell spacing along {} is also non-uniform, so the edge "
                 "placement there is biased by the local stretch ratio.".format(
                     ', '.join('xyz'[d] for d in stretched))), UserWarning)

        ##### Assemble the grid coordinates #####
        # Where a patch covers a face its plane is the domain edge exactly.
        # Where none does, it is inferred from the cell spacing -- the same
        # closure center_cell_regrid uses, which is then handed those resolved
        # edges as `bounds` so the grid and the data cannot disagree.
        self._grid = []
        for d in range(3):
            guess = _infer_domain_edges(interior[d])
            ends = [self._faces[(d, s)][3] if (d, s) in self._faces else guess[s]
                    for s in (0, 1)]
            self._grid.append(np.concatenate(
                ([ends[0]], interior[d], [ends[1]])))
            if not np.all(np.diff(self._grid[-1]) > 0):
                raise RuntimeError(
                    "Boundary patches for the {} direction do not lie outside "
                    "the interior cell centers.".format('xyz'[d]))



    def _read_dump(self, datasets, atol=1e-12, verify=False):
        '''Read one timestep and assemble it onto the full domain grid.

        Returns an (nx+2, ny+2, nz+2, 3) array: the interior block surrounded by
        a shell of boundary values.

        The shell is filled in three stages, because the patch files cover the
        six faces but not the lines and points where faces meet. An edge cell of
        the assembled grid lies on two faces at once and appears in neither
        patch; it is filled from the two faces that meet there, and a corner from
        the three edges. Where the two sides disagree -- a genuine discontinuity
        in the boundary conditions, e.g. an outflow running into a no-slip wall
        -- an exactly zero velocity is taken as no-slip and wins, and anything
        else is averaged. Either way it warns.

        verify additionally checks this dump's cells against the mesh, which
        costs computing the cell centers on top of a read that happens anyway.
        The caller passes it for the second dump of the series; see
        ``_verify_dump_mesh``.
        '''

        nx, ny, nz = self._shape
        vel = np.zeros((nx+2, ny+2, nz+2, 3))

        centers, data, _ = _dataio.read_vtkxml_cell_data(
            datasets['internal'], arrays=('U',),
            load_cell_coordinates=verify)
        U = data['U']
        if len(U) != nx*ny*nz:
            raise RuntimeError(
                "{} has {} cells; the mesh established from the first dump has "
                "{}. The mesh is assumed not to change across the "
                "series.".format(datasets['internal'].name, len(U), nx*ny*nz))
        if verify:
            self._verify_dump_mesh(centers, self._perm, self._shape, (0, 1, 2),
                                   datasets['internal'].name)
        vel[1:-1, 1:-1, 1:-1, :] = U[self._perm].reshape(nx, ny, nz, 3)

        ##### Stage 1: the six faces, from the patch files #####
        patches = {}
        for (d, side), (name, sel, fshape, plane) in self._faces.items():
            if name not in patches:
                if name not in datasets:
                    raise RuntimeError(
                        "Dump {} has no '{}' boundary patch, but the mesh "
                        "established from the first dump needs it for the {} "
                        "face.".format(datasets['internal'].parent.name, name,
                                       'xyz'[d]+'-+'[side]))
                pcenters, pdata, _ = _dataio.read_vtkxml_cell_data(
                    datasets[name], arrays=('U',),
                    load_cell_coordinates=verify)
                # A patch is indexed by the selection built at construction, so
                # one that changed length would take the wrong cells rather than
                # fail. Always checked; it costs a comparison.
                if len(pdata['U']) != self._patch_cells[name]:
                    raise RuntimeError(
                        "Boundary patch {} has {} cells; the mesh established "
                        "from the first dump has {}. The mesh is assumed not "
                        "to change across the series.".format(
                            datasets[name].name, len(pdata['U']),
                            self._patch_cells[name]))
                patches[name] = (pdata['U'], pcenters)
            if verify:
                self._verify_dump_mesh(
                    patches[name][1], sel, fshape,
                    tuple(a for a in range(3) if a != d),
                    datasets[name].name, plane=(d, plane))
            face = patches[name][0][sel].reshape(*fshape, 3)
            idx = [slice(1, -1)]*3
            idx[d] = 0 if side == 0 else -1
            vel[tuple(idx)] = face

        ##### Stage 1b: any face with no patch, by extending the interior #####
        # Only the faces' interior runs are taken; the edges and corners
        # center_cell_regrid also fills are discarded, so that stages 2 and 3
        # below decide those by one rule whatever mix of patched and
        # extrapolated faces meet there. Each face's interior run depends only
        # on the sweep along its own axis, so the discarded parts cannot affect
        # what is kept.
        if len(self._regrid_faces) > 0:
            inner = vel[1:-1, 1:-1, 1:-1, :]
            # bounds= are the edges _build_grid already published, so the field
            # is extended to exactly the grid the loader reports. The two
            # coincide on a uniform mesh whose patches sit half a cell out -- as
            # the reference dataset's do, which is why no test here can tell the
            # difference; center_cell_regrid's own tests pin it.
            ext, _ = center_cell_regrid(
                [inner[..., k] for k in range(3)],
                [g[1:-1] for g in self._grid], self._periodic_dim,
                bounds=[(g[0], g[-1]) for g in self._grid])
            for d, side in self._regrid_faces:
                idx = [slice(1, -1)]*3
                idx[d] = 0 if side == 0 else -1
                vel[tuple(idx)] = np.stack([e[tuple(idx)] for e in ext],
                                           axis=-1)

        ##### Stage 2: the twelve edges, from the two faces meeting there #####
        # Each edge runs along axis c with axes a and b at their extremes. Only
        # the interior run of it is defined by the faces; the two ends are
        # corners, where three faces meet, and are left to stage 3.
        # Where the two faces disagree an exactly zero velocity wins: it marks a
        # no-slip wall, and a wall is no-slip right up to where something else
        # runs into it, so averaging would smear a nonzero velocity onto a
        # surface the fluid cannot move along.
        #
        # Whole vector, exact zero -- no-slip makes every component vanish and
        # the exporter writes 0.0, whereas one component vanishing is ordinary
        # (w = 0 on a z-normal inlet plane) and means nothing. A tolerance would
        # misread slow near-wall flow as no-slip.
        disagreement = 0.
        zeroed = 0
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
                        za = ~np.any(va, axis=-1)
                        zb = ~np.any(vb, axis=-1)
                        vel[tuple(idx)] = np.where((za | zb)[..., None], 0.,
                                                   0.5*(va + vb))
                        # Count only where the rule changed the answer: both
                        # zero already averages to zero.
                        zeroed += int(np.count_nonzero(za ^ zb))
                        disagreement = max(disagreement,
                                           float(np.abs(va-vb).max()))

        ##### Stage 3: the eight corners, from the three edges meeting there ###
        # Same rule. A corner sits on all three faces meeting there, so if any
        # of its edges came out no-slip the corner is on that wall too.
        for sx in (0, -1):
            for sy in (0, -1):
                for sz in (0, -1):
                    ix = 1 if sx == 0 else -2
                    iy = 1 if sy == 0 else -2
                    iz = 1 if sz == 0 else -2
                    edges = (vel[ix, sy, sz, :], vel[sx, iy, sz, :],
                             vel[sx, sy, iz, :])
                    if any(not np.any(e) for e in edges):
                        vel[sx, sy, sz, :] = 0.
                    else:
                        vel[sx, sy, sz, :] = sum(edges)/3

        if disagreement > atol and not self._warned_bc_corner:
            self._warned_bc_corner = True
            warnings.warn(
                "Boundary patches disagree by up to {:g} where they meet along "
                "the edges of the domain. This is a discontinuity in the "
                "boundary conditions themselves (an inflow meeting a no-slip "
                "wall, say), not an error in the data. Where one side is "
                "exactly zero it is taken as no-slip and wins, which is the "
                "case for {} of those cells; the rest are the average of the "
                "two faces.".format(disagreement, zeroed), UserWarning)

        return vel


