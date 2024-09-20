'''Utility functions for calculating geometric quantities. These are the 
workhorses of agent interaction with immersed boundaries. Prior to 0.7, they 
were static methods of swarm.

Created: Thurs September 19 2024

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

__author__ = "Christopher Strickland"
__email__ = "cstric12@utk.edu"
__copyright__ = "Copyright 2017, Christopher Strickland"

import numpy as np
# from scipy.spatial import distance # used in _project_and_slide
    
def closest_dist_btwn_two_lines(a0,a1,b0,b1):
    ''' Given two 3D lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the distance of the closest points on each segment. b0 and b1 
        can be arrays of points; each successive line segment (b0,b1) will 
        be compared to the same (a0,a1).

        Acknowledgement: This code is based on an algorithm for one pair of 
        points by stackoverflow user Fnord, edited by Phil Dukhov. It has 
        been heavily altered here for our use case scenario, vectorized, and 
        its mathematical justification has been checked and documented.

        Assuming the lines formed by (a1-a0) and (b1-b0) are skew, the 
        closest points on the two lines are found using a method described 
        in docs/notes/Line_closest_points.md

        Parameters
        ----------
        a0 : length 2 or length 3 ndarray
            start point for segment A
        a1 : length 2 or length 3 ndarray
            end point for segment A
        b0 : Nx2 or Nx3 ndarray 
            first points in a list of line segments
        b1 : Nx2 or Nx3 ndarray 
            second points in a list of line segments.

        Returns
        -------
        Length N ndarray of distances
    '''
    if len(b0.shape) == 1:
        b0 = np.reshape(b0, (1,len(b0)))
        b1 = np.reshape(b1, (1,len(b1)))

    # Calculate denominator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B, axis=1)
    
    # normalized vectors in the direction of each line
    _A = A / magA
    _B = B / np.tile(magB,(a0.shape[0],1)).T
    
    if a0.shape[0] == 3:
        # 3D
        cross = np.cross(_A, _B)
        denom = np.linalg.norm(cross, axis=1)**2
    else:
        # 2D
        # get a stack of matrices [_A,_B]
        detstack = np.empty((b0.shape[0],2,2))
        detstack[:,0,:] = np.broadcast_to(_A, (b0.shape[0],2))
        detstack[:,1,:] = _B
        # get abs(det([_A,_B]))
        denom = np.linalg.det(detstack)

    # Try to catch machine zeros
    denom_bool = np.abs(denom) < np.finfo(float).eps * 100
    denom[denom_bool] = 0
    
    dist_vals = np.empty(b0.shape[0])

    zero_bool = np.logical_not(denom)
    nonzero_bool = np.logical_not(zero_bool)
    # Find intersection of skew lines and clamp to endpoints as needed.
    #   Then find distance.
    if nonzero_bool.any():
        b0nz = b0[nonzero_bool,:]
        b1nz = b1[nonzero_bool,:]
        magBnz = magB[nonzero_bool]
        _Bnz = _B[nonzero_bool,:]
        if a0.shape[0] == 3:
            # 3D
            t = (b0nz - a0)
            crossnz = cross[nonzero_bool,:]

            # detA = np.linalg.det([t, _B, cross])
            detstack = np.empty((b0nz.shape[0],3,3))
            detstack[:,0,:] = t
            detstack[:,1,:] = _Bnz
            detstack[:,2,:] = crossnz
            detA = np.linalg.det(detstack)
            # detB = np.linalg.det([t, _A, cross])
            detstack[:,1,:] = np.broadcast_to(_A, (b0nz.shape[0],3))
            detB = np.linalg.det(detstack)

            t0 = detA/denom[nonzero_bool]
            t1 = detB/denom[nonzero_bool]
        else:
            # 2D (sol via setting eqn. of two lines equal to each other)
            t = (a0 - b0nz)
            # t0 = np.linalg.det([_B, t])/denom
            detstack = np.empty((b0nz.shape[0],2,2))
            detstack[:,0,:] = _Bnz
            detstack[:,1,:] = t
            detA = np.linalg.det(detstack)
            # t1 = np.linalg.det([_A, t])/denom
            detstack[:,0,:] = np.broadcast_to(_A, (b0nz.shape[0],2))
            detB = np.linalg.det(detstack)

            t0 = detA/denom[nonzero_bool]
            t1 = detB/denom[nonzero_bool]

        # If any determinants come out close to zero, overwrite the values 
        #   calculated here in the parallel section. This should take care 
        #   of anything that the machine epsilon catch before didn't.
        detA_bool = np.abs(detA) < np.finfo(float).eps * 100
        detB_bool = np.abs(detB) < np.finfo(float).eps * 100 
        det_bool = np.logical_or(detA_bool, detB_bool)
        zero_bool[nonzero_bool] = det_bool

        # Projected closest point on line A
        pA = a0 + (np.tile(_A,(len(t0),1)) * np.tile(t0,(a0.shape[0],1)).T)
        # Projected closest point on line B
        pB = b0nz + (_Bnz * np.tile(t1,(a0.shape[0],1)).T) 

        # Clamp projections
        if len(pA.shape) == 1: # check for only one point (a0 not prev. reshaped)
            pA = np.reshape(pA, (1,len(pA)))
        pA[t0 < 0,:] = a0
        pA[t0 > magA,:] = a1
        
        pB[t1 < 0,:] = b0nz[t1 < 0]
        pB[t1 > magBnz,:] = b1nz[t1 > magBnz]
            
        # Clamp projection A
        past_A_bool = np.logical_or(t0 < 0, t0 > magA)
        if past_A_bool.any():
            # dot = np.dot(_B,(pA-b0))
            dot = (_Bnz[past_A_bool,:]*(pA[past_A_bool,:]-b0nz[past_A_bool,:])).sum(1)
            dot = np.maximum(dot, 0)
            dot = np.minimum(dot, magBnz[past_A_bool])
            pB[past_A_bool,:] = b0nz[past_A_bool,:] + (_Bnz[past_A_bool,:] 
                                                        * np.tile(dot,(a0.shape[0],1)).T)
    
        # Clamp projection B
        past_B_bool = np.logical_or(t1 < 0, t1 > magBnz)
        if past_B_bool.any():
            # dot = np.dot(_A,(pB-a0))
            dot = np.inner(_A,(pB[past_B_bool,:]-a0))
            dot = np.maximum(dot, 0)
            dot = np.minimum(dot, magA)
            pA[past_B_bool,:] = a0 + (_A * np.tile(dot,(a0.shape[0],1)).T)

        dist_vals[nonzero_bool] = np.linalg.norm(pA-pB, axis=1)


    # If lines are parallel (denom=0) test if segment projections overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there 
    #   is a closest distance
    # This case happens more rarely, so treat in a for-loop
    if zero_bool.any():
        b0_zero = b0[zero_bool,:]
        b1_zero = b1[zero_bool,:]
        return_vals = np.empty(b0_zero.shape[0])

        n = 0
        for b0z, b1z in zip(b0_zero,b1_zero):

            d0 = np.inner(_A,(b0z-a0))
            d1 = np.inner(_A,(b1z-a0))
                
            # Is segment B completely "before" A?
            if d0 <= 0 and d1 <= 0:
                # Then the shortest distance is between whatever endpoint of b
                #   is closer to a0
                if np.abs(d0) < np.abs(d1):
                    return_vals[n] =  np.linalg.norm(a0-b0z)
                else:
                    return_vals[n] = np.linalg.norm(a0-b1z)
                
            # Is segment B completely "after" A?
            elif d0 >= magA and d1 >= magA:
                # Then the shortest distance is between whatever endpoint of b
                #   is closer to a1
                if np.abs(d0) < np.abs(d1):
                    return_vals[n] = np.linalg.norm(a1-b0z)
                else:
                    return_vals[n] = np.linalg.norm(a1-b1z)
                    
            # The projection of the segments overlap. Return distance between 
            #   parallel segments (closest points are not unique).
            else:
                # Translate a0 along _A until it is perpendicular to b0, then 
                #   find dist.
                return_vals[n] = np.linalg.norm(((d0*_A)+a0)-b0z)

            n += 1
        dist_vals[zero_bool] = return_vals

    return dist_vals



def seg_intersect_2D(P0, P1, Q0_list, Q1_list, get_all=False):
    '''Find the intersection between two line segments (2D objects), P and Q, 
    returning None if there isn't one or if they are parallel.

    If Q is a 2D array, loop over the rows of Q finding all intersections
    between P and each row of Q, but only return the closest intersection
    to P0 (if there is one, otherwise None)

    This works for both 2D problems and problems in which P is a 3D line 
    segment roughly lying on a plane (e.g., in cases of projection along a 
    3D triangular mesh element). The plane is described by the first two 
    vectors Q, so in this case, Q0_list and Q1_list must have at least two 
    rows. The 3D problem is robust to cases where P is not exactly in the 
    plane because the algorithm is actually checking to see if its 
    projection onto the triangle crosses any of the lines in Q. This is 
    important to deal with roundoff error.

    This algorithm uses a parameteric equation approach for speed, based on
    [1]_
    
    Parameters
    ----------
    P0 : length 2 (or 3) array
        first point in line segment P
    P1 : length 2 (or 3) array
        second point in line segment P 
    Q0_list : Nx2 (Nx3) ndarray 
        first points in a list of line segments.
    Q1_list : Nx2 (Nx3) ndarray 
        second points in a list of line segments.
    get_all : bool
        Return all intersections instead of just the first one encountered 
        as one travels from P0 to P1.

    Returns
    -------
    None if there is no intersection. Otherwise:
    x : length 2 (or 3) array 
        the coordinates of the point of first intersection
    s_I : float between 0 and 1
        the fraction of the line segment traveled from P0 to P1 before
        intersection occurred
    vec : length 2 (or 3) array
        directional unit vector along the boundary (Q) intersected
    Q0 : length 2 (or 3) array
        first endpoint of mesh segment intersected
    Q1 : length 2 (or 3) array
        second endpoint of mesh segment intersected

    References
    ----------
    .. [1] Sunday, Daniel, (2021). Practial Geometry Algorithms with C++ 
        Code, self-published: Amazon KDP.
    '''

    u = P1 - P0
    v = Q1_list - Q0_list
    w = P0 - Q0_list

    if len(P0) == 2:
        u_perp = np.array([-u[1], u[0]])
        v_perp = np.array([-v[...,1], v[...,0]]).T
    else:
        normal = np.cross(v[0],v[1])
        normal /= np.linalg.norm(normal)
        # roundoff error in only an np.dot projection can be as high as 1e-7
        assert np.isclose(np.dot(u,normal),0,atol=1e-6), "P vector not parallel to Q plane"
        u_perp = np.cross(u,normal)
        v_perp = np.cross(v,normal)


    if len(Q0_list.shape) == 1:
        # only one point in Q list
        denom = np.dot(v_perp,u)
        if denom != 0:
            s_I = np.dot(-v_perp,w)/denom
            t_I = -np.dot(u_perp,w)/denom
            if 0<=s_I<=1 and 0<=t_I<=1:
                return (P0 + s_I*u, s_I, v/np.linalg.norm(v), Q0_list, Q1_list)
        return None

    denom_list = np.multiply(v_perp,u).sum(1) #vectorized dot product

    # We need to deal with parallel cases. With roundoff error, exact zeros
    #   are unlikely (but ruled out below). Another possiblity is getting
    #   inf values, but these will not record as being between 0 and 1, and
    #   will not throw errors when compared to these values. So all should
    #   be good.
    # 
    # Non-parallel cases:
    not_par = denom_list != 0

    # All non-parallel lines in the same plane intersect at some point.
    #   Find the parametric values for both vectors at which that intersect
    #   happens. Call these s_I and t_I respectively.
    s_I_list = -np.ones_like(denom_list)
    t_I_list = -np.ones_like(denom_list)
    # Now only need to calculuate s_I & t_I for non parallel cases; others
    #   will report as not intersecting automatically (as -1)
    #   (einsum is faster for vectorized dot product, but need same length,
    #   non-empty vectors)
    # In 3D, we are actually projecting u onto the plane of the triangle
    #   and doing our calculation there. So no problem about numerical
    #   error and offsets putting u above the plane.
    if np.any(not_par):
        s_I_list[not_par] = np.einsum('ij,ij->i',-v_perp[not_par],w[not_par])/denom_list[not_par]
        t_I_list[not_par] = -np.multiply(u_perp,w[not_par]).sum(1)/denom_list[not_par]

    # The length of our vectors parameterizing the lines is the same as the
    #   length of the line segment. So for the line segments to have intersected,
    #   the parameter values for their intersect must both be in the unit interval.
    intersect = np.logical_and(
                    np.logical_and(0<=s_I_list, s_I_list<=1),
                    np.logical_and(0<=t_I_list, t_I_list<=1))

    if np.any(intersect):
        if get_all:
            x = []
            for s_I in s_I_list[intersect]:
                x.append(P0+s_I*u)
            return zip(
                x, s_I_list[intersect], 
                v[intersect]/np.linalg.norm(v[intersect]),
                Q0_list[intersect], Q1_list[intersect]
            )
        else:
            # find the closest intersection and return it
            Q0 = Q0_list[intersect]
            Q1 = Q1_list[intersect]
            v_intersected = v[intersect]
            s_I = s_I_list[intersect].min()
            s_I_idx = s_I_list[intersect].argmin()
            return (P0 + s_I*u, s_I,
                    v_intersected[s_I_idx]/np.linalg.norm(v_intersected[s_I_idx]),
                    Q0[s_I_idx], Q1[s_I_idx])
    else:
        return None



def seg_intersect_3D_triangles(P0, P1, Q0_list, Q1_list, Q2_list, get_all=False):
    '''Find the intersection between a line segment P0 to P1 and any of the
    triangles given by Q0, Q1, Q2 where each row across the three arrays is 
    a different triangle (three points).
    Returns None if there is no intersection.

    This algorithm uses a parameteric equation approach for speed, based on
    [2]_

    Parameters
    ----------
    P0 : length 3 array
        first point in line segment P
    P1 : length 3 array
        second point in line segment P 
    Q0 : Nx3 ndarray 
        first points in a list of triangles.
    Q1 : Nx3 ndarray 
        second points in a list of triangles.
    Q2 : Nx3 ndarray 
        third points in a list of triangles.
    get_all : bool
        Return all intersections instead of just the first one encountered 
        as you travel from P0 to P1.

    Returns
    -------
    None if there is no intersection. Otherwise: 
    x : length 3 array 
        the coordinates of the first point of intersection
    s_I : float between 0 and 1
        the fraction of the line segment traveled from P0 before 
        intersection occurred (only if intersection occurred)
    normal : length 3 array
        normal unit vector to plane of intersection
    Q0 : length 3 array
        first vertex of triangle intersected
    Q1 : length 3 array
        second vertex of triangle intersected
    Q2 : length 3 array
        third vertex of triangle intersected
    
    References
    ----------
    .. [1] Sunday, Daniel, (2021). Practial Geometry Algorithms with C++ 
        Code, self-published: Amazon KDP.
    '''

    Q1Q0_diff = Q1_list-Q0_list
    Q2Q0_diff = Q2_list-Q0_list
    n_list = np.cross(Q1Q0_diff, Q2Q0_diff)

    u = P1 - P0
    w = P0 - Q0_list

    # First, determine intersections between the line segment and full planes
    s_I_list = seg_intersect_3D_plane(u, n_list, w)

    if len(Q0_list.shape) == 1:
        # only one triangle
        if s_I_list is None:
            return None
        else:
            cross_pt = P0 + s_I_list*u
            # calculate barycentric coordinates
            normal = n_list/np.linalg.norm(n_list)
            A_dbl = np.dot(n_list, normal)
            Q0Pt = cross_pt-Q0_list
            A_u_dbl = np.dot(np.cross(Q0Pt,Q2Q0_diff),normal)
            A_v_dbl = np.dot(np.cross(Q1Q0_diff,Q0Pt),normal)
            coords = np.array([A_u_dbl/A_dbl, A_v_dbl/A_dbl, 0])
            coords[2] = 1 - coords[0] - coords[1]
            # check if point is in triangle
            if np.all(coords>=0):
                return (cross_pt, s_I_list, normal, Q0_list, Q1_list, Q2_list)
            else:
                return None


    # calculate barycentric coordinates for each plane intersection
    closest_int = (None, -1, None)
    intersections = []
    for n, s_I in zip(np.arange(len(s_I_list))[s_I_list!=-1], s_I_list[s_I_list!=-1]):
        # if get_all is False, we only care about the closest triangle intersection!
        # see if we need to worry about this one
        if closest_int[1] == -1 or closest_int[1] > s_I or get_all:
            cross_pt = P0 + s_I*u
            normal = n_list[n]/np.linalg.norm(n_list[n])
            A_dbl = np.dot(n_list[n], normal)
            Q0Pt = cross_pt-Q0_list[n]
            A_u_dbl = np.dot(np.cross(Q0Pt,Q2Q0_diff[n]),normal)
            A_v_dbl = np.dot(np.cross(Q1Q0_diff[n],Q0Pt),normal)
            coords = np.array([A_u_dbl/A_dbl, A_v_dbl/A_dbl, 0])
            coords[2] = 1 - coords[0] - coords[1]
            # check if point is in triangle
            if np.all(coords>=0):
                if get_all:
                    intersections.append((cross_pt, s_I, normal, Q0_list[n], Q1_list[n], Q2_list[n]))
                else:
                    closest_int = (cross_pt, s_I, normal, Q0_list[n], Q1_list[n], Q2_list[n])
    if not get_all:
        if closest_int[0] is None:
            return None
        else:
            return closest_int
    else:
        if len(intersections) == 0:
            return None
        else:
            return intersections



def seg_intersect_3D_plane(u, n_list, w):
    '''Given a 3D line segment from P0 to P1 in 3D, determine the intersection 
    between the line segment and each plane defined by three points Q0, Q1, Q2. 
    It is assumed that while the line segment may be parallel to the plane, it 
    does not lie perfectly within the plane itself.

    Parameters
    ----------
    u : vector P1 - P0 of line segment
    n_list : Nx3 ndarray 
        normal vectors to the planes
    w : Nx3 ndarray
        P0 - Q0 vectors from first point in line segment to a point on each plane
        
    Returns
    -------
    - If there is only one plane and no intersection, None is returned
    - If there is one plane with an intersection a float in [0,1] is returned
        corresponding to the fraction of the line segment traveled from P0 
        before intersection occurred
    - If there are multiple planes, a length N array of floats is returned.
        -1 is coded to mean that the line segment did not intersect with that 
        plane. Otherwise, the faction of the line segment traveled from P0 
        before intersection occurred with that plane is recorded

    References
    ----------
    .. [1] Sunday, Daniel, (2021). Practial Geometry Algorithms with C++ 
        Code, self-published: Amazon KDP.
    '''

    # At intersection, w + su is perpendicular to n

    ##### Only one Q plane case #####
    if len(w.shape) == 1:
        # only one plane
        denom = np.dot(n_list,u)
        if denom != 0:
            s_I = np.dot(-n_list,w)/denom
            if 0<=s_I<=1:
                # line segment crosses full plane
                return s_I
        return None

    ##### Multiple planes #####
    denom_list = np.multiply(n_list,u).sum(1) #vectorized dot product

    # record non-parallel cases
    not_par = denom_list != 0

    # default is not intersecting (coded as -1)
    s_I_list = -np.ones_like(denom_list)
    
    # get intersection parameters
    #   (einsum is faster for vectorized dot product, but need same length vectors)
    if np.any(not_par):
        s_I_list[not_par] = np.einsum('ij,ij->i',-n_list[not_par],w[not_par])/denom_list[not_par]
    # test for intersection of line segment with full plane.
    #   Reset s_I_list to -1 for all non-intersecting cases.
    s_I_list[np.logical_or(s_I_list<0, s_I_list>1)] = -1
    # plane_int = np.logical_and(0<=s_I_list, s_I_list<=1)

    return s_I_list



def seg_intersect_3D_quadrilateral(P0, P1, Q0_list, Q1_list, Q2_list, 
                                       Q3_list, get_all=False):
    '''Find the intersection between a 2D line segment and quadrilaterals 
    in the 3D space formed by two spatial dimensions and time for the 
    application of finding intersections between a moving point and a moving 
    1D mesh element in 2D space and time. All arguments are assumed to be 
    given as 2D spatial points.
    
    Since dt is the same for both the line segment and the moving mesh 
    elements, it can be assumed that P0, Q0, and Q1 are coplaner in the 
    t-dimension, and similarly for P1, Q2, and Q3. The t-direction is 
    therefore normalized to 0 and 1.

    TODO: Testing needed!
    
    Parameters
    ----------
    P0 : length 2 array
        first point in line segment P
    P1 : length 2 array
        second point in line segment P 
    Q0_list : Nx2 ndarray 
        first points in a list of 2D mesh elements at starting time.
    Q1_list : Nx2 ndarray 
        second points in a list of 2D mesh elements at starting time.
    Q2_list : Nx2 ndarray 
        first points in a list of 2D mesh elements at ending time.
    Q3_list : Nx2 ndarray
        second points in a list of 2D mesh elements at ending time.
    get_all : bool
        Return all intersections instead of just the first one encountered 
        as you travel from P0 to P1.

    Returns
    -------
    None if there is no intersection. Otherwise: 
    x : length 2 array 
        the (x,y) coordinates of the first point of intersection
    s_I : float between 0 and 1
        the fraction of the line segment traveled from P0 before 
        intersection occurred (only if intersection occurred)
    vec : length 2 array
        unit vector in the direction Q1 - Q0 at time of intersection
    Q0 : length 3 array
        first point of mesh element that was intersected at time 0
    Q1 : length 3 array
        second point of mesh element that was intersected at time 0
    Q2 : length 3 array
        first point of mesh element that was intersected at time dt
    Q3 : length 3 array
        second point of mesh element that was intersected at time dt
    '''
    
    # Get vectors normal to each plane leveraging unit-length in t direction
    Q1Q0_diff = Q1_list-Q0_list
    Q3Q2_diff = Q3_list-Q2_list

    u = P1 - P0
    w = P0 - Q0_list
    
    if len(Q0_list.shape) == 1:
        # Only one plane
        u = np.hstack((P1-P0,1))
        w = np.hstack((P0-Q0_list,0))
        # cross product
        n_list = np.array([Q1Q0_diff[1], -Q1Q0_diff[0], 
                           np.linalg.det(np.array([Q1Q0_diff,Q3Q2_diff]))])
    else:
        u = np.hstack((P1-P0,np.ones((Q0_list.shape[0],1))))
        w = np.hstack((P0-Q0_list,np.zeros((Q0_list.shape[0],1))))
        n_list = np.empty((Q0_list.shape[0],3))
        # cross product
        n_list[:,0] = Q1Q0_diff[:,1]
        n_list[:,1] = -Q1Q0_diff[:,0]
        n_list[:,2] = Q1Q0_diff[:,0]*Q3Q2_diff[:,1] - Q1Q0_diff[:,1]*Q3Q2_diff[:,0]

    # determine intersections between line segement and full planes
    s_I_list = seg_intersect_3D_plane(u, n_list, w)

    ##### Narrow down to quadrilaterals #####

    if len(Q0_list.shape) == 1:
        # Single plane case
        if s_I_list is None:
            return None
        else:
            cross_pt = P0 + s_I_list*u
            # Check for intersections outside of t unit interval
            if cross_pt[2] < 0 or cross_pt[2] > 1:
                return None
            # Check if intersection is within mesh element. To do this, find the 
            #   mesh element vertices at the time of intersection and see if the 
            #   point of intersectin is between them.
            # We know the pt of intersection is colinear, so we just need to 
            #   compare to endpoints. A one dimensional check is enough except 
            #   when the mesh element lies in the x- or y-direction. So check 
            #   both to be safe.
            first_pt = Q0_list + (Q2_list-Q0_list)*s_I_list
            second_pt = Q1_list + (Q3_list-Q1_list)*s_I_list
            if np.all(np.logical_and(first_pt <= cross_pt[:2], 
                                     cross_pt[:2] <= second_pt)):
                return (cross_pt, s_I_list, 
                        (second_pt-first_pt)/np.linalg.norm(second_pt-first_pt), 
                        Q0_list, Q1_list, Q2_list, Q3_list)
            else:
                return None
            
    # Multiple plane case
    closest_int = (None, -1)
    intersections = []
    for n, s_I in zip(np.arange(len(s_I_list))[s_I_list!=-1], s_I_list[s_I_list!=-1]):
        # if get_all is False, we only care about the closest intersection!
        # see if we need to worry about each one, and then record as appropriate
        if closest_int[1] == -1 or closest_int[1] > s_I or get_all:
            cross_pt = P0 + s_I*u
            # Check that intersection is inside t unit interval
            if 0 <= cross_pt[2] <= 1:
                # Check if intersection is within mesh element
                first_pt = Q0_list[n] + (Q2_list[n]-Q0_list[n])*s_I
                second_pt = Q1_list[n] + (Q3_list[n]-Q1_list[n])*s_I
                if np.all(np.logical_and(first_pt <= cross_pt[:2], 
                                         cross_pt[:2] <= second_pt)):
                    intersec = (cross_pt, s_I, 
                                (second_pt-first_pt)/np.linalg.norm(second_pt-first_pt),
                                Q0_list[n], Q1_list[n], Q2_list[n], Q3_list[n])
                    if get_all:
                        intersections.append(intersec)
                    else:
                        closest_int = intersec
    if not get_all:
        if closest_int[0] is None:
            return None
        else:
            return closest_int
    else:
        if len(intersections) == 0:
            return None
        else:
            return intersections



def dist_point_to_plane(P0, normal, Q0):
    '''Return the distance from the point P0 to the plane given by a
    normal vector and a point on the plane, Q0. For debugging.'''

    d = np.dot(normal, Q0)
    return np.abs(np.dot(normal,P0)-d)/np.linalg.norm(normal)


