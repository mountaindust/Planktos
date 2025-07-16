'''These are the key functions for handling immersed boundary interactions.
They were formally static methods of the Swarm class.

Created: Thurs July 16 2025

Author: Christopher Strickland

Email: cstric12@utk.edu
'''

import numpy as np
from scipy import integrate, optimize
from . import _geom


def _apply_internal_static_BC(startpt, endpt, mesh, max_meshpt_dist, 
                                ib_collisions='sliding'):
    '''Apply internal boundaries to a trajectory starting and ending at
    startpt and endpt, returning a new endpt (or the original one) as
    appropriate.

    Parameters
    ----------
    startpt : length 2 or 3 array
        start location for agent trajectory
    endpt : length 2 or 3 array
        end location for agent trajectory
    mesh : Nx2x2 or Nx3x3 array 
        eligible mesh elements to check for intersection
    max_meshpt_dist : float
        max distance between two points on a mesh element
        (used to determine how far away from startpt to search for
        mesh elements)
    old_intersection : list-like of data
        (for internal use only) records the last intersection in the 
        recursion to check if we are bouncing back and forth between two 
        boundaries as a result of a concave angle and the right kind of 
        trajectory vector.
    kill : bool
        (for internal use only) set to True in 3D case if we have previously 
        slid along the boundary line between two mesh elements. This 
        prevents such a thing from happening more than once, in case of 
        pathological cases.
    ib_collisions : {'sliding' (default), 'sticky'}
        Type of interaction with immersed boundaries. In sliding 
        collisions, conduct recursive vector projection until the length of
        the original vector is exhausted. In sticky collisions, just return 
        the point of intersection.

    Returns
    -------
    newendpt : length 2 or 3 array
        new end location for agent trajectory
    dx : length 2 or 3 array, or None
        change in position for agent after IB collision - based on first
        collision point and final location. If no IB collision, None.

    Acknowledgements
    ---------------
    Appreciation goes to Anne Ho, for pointing out that the centroid of an
    equalateral triangle is further away from any of its vertices than I had 
    originally assumed it was.
    '''

    if len(startpt) == 2:
        DIM = 2
    else:
        DIM = 3

    # We only want to check mesh elements that could feasibly intersect 
    #   the line segment startpt endpt. Mesh elements are identified by 
    #   their vertices; the question is: how far away from the line segment 
    #   do we need to look for mesh vertices? Part of the answer to this 
    #   question depends on how big mesh elements can be, since the longer 
    #   they are, the further away their vertices could be.

    # In barycentric coordinates, the centoid is always (1/3,1/3,1/3), and
    #   the entries of the coordinates must add to 1. This suggests that the
    #   furthest away you can be from every vertex simultaneously is 2/3 
    #   down the medians (an increase in one barycentric coordinate results 
    #   in a decrease in the others). Since the median length is bounded 
    #   above by the length of the longest side of the triangle, circles 
    #   centered at each vertex that are 2/3 times the length of the longest 
    #   triangle side should be sufficient to cover any triangle.
    # More precisely: equalateral triangles are probably the worst case. If 
    #   so, all medians have length l*sqrt(3/4), where l is the length of a 
    #   side of the triangle. This implies circles of radius l*sqrt(3/4)*2/3
    #   are a strict lower bound on covering any circle.

    # The result of this argument, from the worst-case scenario equalateral 
    #   triangle in our collection, forms the radius we need to search from 
    #   the line segment of travel in order to find vertices of mesh elements
    #   that we potentially intersected.
    search_rad = max_meshpt_dist*2/3

    # Find all mesh elements that have vertex points within search_rad of 
    #   the trajectory segment.
    close_mesh = _get_eligible_static_mesh_elements(startpt, endpt, mesh, 
                                                            search_rad)

    # Get intersections
    if DIM == 2:
        intersection = _geom.seg_intersect_2D(startpt, endpt,
            close_mesh[:,0,:], close_mesh[:,1,:])
    else:
        intersection = _geom.seg_intersect_3D_triangles(startpt, endpt,
            close_mesh[:,0,:], close_mesh[:,1,:], close_mesh[:,2,:])

    # Return endpt we already have if None.
    if intersection is None:
        return endpt, None
    
    # If we do have an intersection:
    if ib_collisions == 'sliding':
        # Get eligible mesh elements for full range of projective travel

        # Get elements out of close_mesh into list for concatenation
        elems = [elem for elem in close_mesh]
        # Get new elements
        search_rad = np.linalg.norm(endpt-intersection[0])/2 + max_meshpt_dist
        # pull all mesh elements with either vertex within a radius of 
        #   search_rad of the halfway point between intersection and endpt
        midpt = intersection[0] + (endpt-intersection[0])/2
        close_elems = mesh[np.linalg.norm(mesh-midpt, axis=2).min(axis=1)<search_rad]
        # take only the elements that are not already in close_mesh and add 
        #   them to close_mesh
        new_elems = [x for x in close_elems if not np.any((x == close_mesh).all(axis=(1,2)))]
        # concatenate. THIS MAINTAINS IDX FROM INTERSECTION OBJECT.
        close_mesh = np.stack(elems+new_elems)

        # Project remaining piece of vector onto mesh and repeat processes 
        #   as necessary until we have a final result.
        new_pos = _project_and_slide_static(startpt, endpt, intersection, 
                                                    close_mesh, max_meshpt_dist)
        return new_pos, new_pos - intersection[0]
    
    elif ib_collisions == 'sticky':
        # Return the point of intersection
        
        # small number to perturb off of the actual boundary in order to avoid
        #   roundoff errors that would allow penetration
        # base its magnitude off of the given coordinate points
        coord_mag = np.ceil(np.log(np.max(
            np.concatenate((startpt,endpt,close_mesh),axis=None))))
        EPS = 10**(coord_mag-7)

        back_vec = (startpt-endpt)/np.linalg.norm(endpt-startpt)
        return intersection[0] + back_vec*EPS, np.zeros((DIM))



def _get_eligible_static_mesh_elements(startpt, endpt, mesh, search_rad):
    '''
    From a list of mesh elements (mesh), find all elements that have vertex 
    points within search_rad of the trajectory segment startpt,endpt.
    '''
    
    pt_bool = _geom.closest_dist_btwn_line_and_pts(startpt, endpt, 
        mesh.reshape((mesh.shape[0]*mesh.shape[1],mesh.shape[2])))<=search_rad
    pt_bool = pt_bool.reshape((mesh.shape[0],mesh.shape[1]))
    return mesh[np.any(pt_bool,axis=1)]
    


def _apply_internal_moving_BC(startpt, endpt, start_mesh, end_mesh, 
                                max_meshpt_dist, max_mov, 
                                ib_collisions='sliding'):
    '''Apply internal boundaries to a trajectory starting and ending at
    startpt and endpt, returning a new endpt (or the original one) as
    appropriate.

    Parameters
    ----------
    startpt : length 2 or 3 array
        start location for agent trajectory
    endpt : length 2 or 3 array
        end location for agent trajectory
    start_mesh : Nx2x2 or Nx3x3 array
        starting position for the mesh
    end_mesh : Nx2x2 or Nx3x3 array
        ending position for the mesh
    max_meshpt_dist : float
        maximum distance between two mesh vertices at either time
    max_mov : float
        maximum distance any mesh vertex moved
    ib_collisions : {'sliding' (default), 'sticky'}
        Type of interaction with immersed boundaries. In sliding 
        collisions, conduct recursive vector projection until the length of
        the original vector is exhausted. In sticky collisions, just return 
        the point of intersection.

    Returns
    -------
    newendpt : length 2 or 3 array
        new end location for agent trajectory
    dx : length 2 or 3 array, or None
        change in position for agent after IB collision - based on first
        collision point and final location. If no IB collision, None.
    '''

    if len(startpt) == 2:
        DIM = 2
    else:
        DIM = 3

    # See static case for derivation of the 2/3 argument
    search_rad = max_meshpt_dist*2/3

    close_mesh_start, close_mesh_end = \
        _get_eligible_moving_mesh_elements(startpt, endpt, start_mesh, 
                                                    end_mesh, max_mov, search_rad)
    
    # Get intersections
    if DIM == 2:
        # find interesections between line segment of motion and 
        #   quadrilateral in 3D (t,x,y) space
        intersection = _geom.seg_intersect_2D_multilinear_poly(startpt, endpt,
                            close_mesh_start[:,0,:], close_mesh_start[:,1,:],
                            close_mesh_end[:,0,:], close_mesh_end[:,1,:])
    else:
        # find interesections between line segment of motion and 
        #   hyper-quadrilateral (hyper-plane) in 4D (t,x,y,z) space
        raise NotImplementedError("3D moving meshes not currently supported.")
    

    # Return endpt we already have if None.
    if intersection is None:
        return endpt, None
    
    # If we have an intersection with this agent, apply boundary condition
    if ib_collisions == 'sticky':
        # Return the point of intersection

        # small number to perturb off of the actual boundary in order to avoid
        #   roundoff errors that would allow boundary penetration
        # base its magnitude off of the given coordinate points
        coord_mag = np.ceil(np.log(np.max(
            np.concatenate((startpt,endpt,close_mesh_start,close_mesh_end),axis=None))))
        EPS = 10**(coord_mag-7)

        if DIM == 2:
            x = intersection[0]    # (x,y) coordinates of intersection
            t_I = intersection[1]
            Q0 = intersection[2]   # edge of mesh element at time of intersection
            Q1 = intersection[3]   # edge of mesh element at time of intersection
            idx = intersection[4]  # index into close_mesh_start/end above

            # Vector for agent travel
            vec = endpt - startpt

            # Get the relative position of intersection within the mesh element
            # Use max in case the element is vertical or horizontal
            s_I = max((x[0]-Q0[0])/(Q1[0]-Q0[0]),(x[1]-Q0[1])/(Q1[1]-Q0[1]))
            if idx is None:
                dt_elem = close_mesh_end
            else:
                dt_elem = close_mesh_end[idx]
            # Translate to final position of mesh element in this time step
            new_pos = dt_elem[0,:]*(1-s_I)+ dt_elem[1,:]*s_I
            
            # Perturb a small bit off of the boundary.
            #   This needs to be on the side of the element the motion 
            #   came from.
            # Find direction indicator for which side of the mesh element 
            #   the agent hit the element. Base this off of location of mesh 
            #   and agent a small time before intersection.
            tm = t_I - 0.00001
            agent_prev_loc = startpt + vec*tm
            Q_tI = Q1-Q0 # vector in direction of mesh elem at t_I
            # vec in dir of mesh element at tm
            Qvec_m = ((1-tm)*Q_tI+(tm-t_I)*(dt_elem[1]-dt_elem[0]))/(1-t_I)
            # interp of Q0 at tm
            Q0_tm = ((1-tm)*Q0+(tm-t_I)*dt_elem[0])/(1-t_I)
            x_prev_loc = s_I*Qvec_m + Q0_tm
            dir_vec = agent_prev_loc-x_prev_loc
            Q_tI_orth = np.array([Q_tI[1],-Q_tI[0]])
            side_signum = np.dot(dir_vec,Q_tI_orth)\
                /np.linalg.norm(np.dot(dir_vec,Q_tI_orth))

            # Now, perturb perpendicular from the end position of the element
            perp_vec = np.array([dt_elem[1,1]-dt_elem[0,1],dt_elem[0,0]-dt_elem[1,0]])
            perp_vec *= side_signum/np.linalg.norm(perp_vec)
            return new_pos + perp_vec*EPS, new_pos - x
        else:
            raise NotImplementedError("3D moving meshes not currently supported.")
    else:
        # Get eligible mesh elements for full range of projective travel
        #   This is going to be really coarse.

        # Get elements out of close_mesh into list for concatenation
        elems_start = [elem for elem in close_mesh_start]
        elems_end = [elem for elem in close_mesh_end]
        # Get new elements
        search_rad = np.linalg.norm(endpt-intersection[0])/2 + max_meshpt_dist + max_mov
        # pull all mesh elements with either vertex within a radius of 
        #   search_rad of the halfway point between intersection and endpt
        midpt = intersection[0] + (endpt-intersection[0])/2
        # base selection off of end_mesh positions
        mesh_bool = np.linalg.norm(end_mesh-midpt, axis=2).min(axis=1)<search_rad
        close_elems_start = start_mesh[mesh_bool]
        close_elems_end = end_mesh[mesh_bool]
        # take only the elements that are not already in close_mesh and add 
        #   them to close_mesh
        new_elems_start = [x for x in close_elems_start 
                            if not np.any((x == close_mesh_start).all(axis=(1,2)))]
        new_elems_end = [x for x in close_elems_end 
                            if not np.any((x == close_mesh_end).all(axis=(1,2)))]
        # concatenate. THIS MAINTAINS IDX FROM INTERSECTION OBJECT.
        close_mesh_start = np.stack(elems_start+new_elems_start)
        close_mesh_end = np.stack(elems_end+new_elems_end)
        
        if DIM == 2:
            # Project remaining piece of vector onto mesh and repeat processes 
            #   as necessary until we have a final result.
            new_pos = _project_and_slide_moving(startpt, endpt, intersection, 
                                            close_mesh_start, close_mesh_end, 
                                            max_meshpt_dist, max_mov)
            return new_pos, new_pos - intersection[0]

        else:
            raise NotImplementedError("3D moving meshes not currently supported.")
            


def _get_eligible_moving_mesh_elements(startpt, endpt, start_mesh, end_mesh, 
                                        max_mov, search_rad):
    '''
    From starting and ending points for the mesh, find all elements that 
    have vertex points which passed within search_rad of the trajectory 
    segment startpt,endpt. Return a tuple of the start and end eligible meshes.
    '''

    #Unfortunately, finding the closest distance between two lines is likely 
    #   slower than just finding the distance between a line and a point as 
    #   in the static case. Instead, do a coarse rule-out and then refine 
    #   with closest distance between two lines.

    # 1/2 a distance between two points is the furthest away you can be and 
    #   still intersect the line segment between them
    outer_rad = search_rad + 0.5*np.linalg.norm(endpt-startpt) + 0.5*max_mov
    start_mesh_r = start_mesh.reshape((start_mesh.shape[0]*start_mesh.shape[1],
                                        start_mesh.shape[2]))
    end_mesh_r = end_mesh.reshape((end_mesh.shape[0]*end_mesh.shape[1],
                                    end_mesh.shape[2]))
    dist_array = np.empty((4, start_mesh_r.shape[0]))
    dist_array[0,:] = np.linalg.norm(startpt - start_mesh_r, axis=1) < outer_rad
    dist_array[1,:] = np.linalg.norm(startpt - end_mesh_r, axis=1) < outer_rad
    dist_array[2,:] = np.linalg.norm(endpt - start_mesh_r, axis=1) < outer_rad
    dist_array[3,:] = np.linalg.norm(endpt - end_mesh_r, axis=1) < outer_rad
    outer_bool = np.any(dist_array, axis=0)

    # anything within the outer radius gets a better check
    dist_list = _geom.closest_dist_btwn_two_lines(startpt, endpt,
        start_mesh_r[outer_bool,:], end_mesh_r[outer_bool,:])
    inner_bool = dist_list < search_rad

    # refine outer_bool with inner_bool
    outer_bool[outer_bool] = inner_bool

    pt_bool = outer_bool.reshape((start_mesh.shape[0], start_mesh.shape[1]))
    return (start_mesh[np.any(pt_bool,axis=1)], end_mesh[np.any(pt_bool,axis=1)])



def _project_and_slide_static(startpt, endpt, intersection, mesh, 
                                max_meshpt_dist, prev_idx=None):
    '''Once we have an intersection point with an immersed mesh, slide the 
    agent along the mesh for its remaining movement (frictionless 
    boundary interaction), and then determine what happens if we fall off 
    the edge of the element or if the element rotates away.
    
    NOTE: Because we do not know the mass of the agents or the properties
    of the fluid, we neglect inertial and drag forces in this computation.

    Parameters
    ----------
    startpt : length 2 or 3 array
        original start point of movement, before intersection
    endpt : length 2 or 3 array
        original end point of movement, w/o intersection
    intersection : list-like of data
        result of seg_intersect_2D or seg_intersect_3D_triangles. various 
        information about the intersection with the immersed mesh element
    mesh : Nx2x2 or Nx3x3 array
        eligible (nearby) mesh elements for interaction
    max_meshpt_dist : float
        max distance between two points on a mesh element (used to determine 
        how far away from startpt to search for mesh elements). 
        Passed-through here solely in case of recursion with 
        _apply_internal_moving_BC.
    prev_idx : int, optional
        in recursion with adjoining mesh elements, this prevents an infinite 
        recusion with, e.g., mesh elements in an acute angle.

    Returns
    -------
    newendpt : length 2 or 3 array
        new endpoint for movement after projection
    '''

    # small number to perturb off of the actual boundary in order to avoid
    #   roundoff errors that would allow penetration
    # base its magnitude off of the given coordinate points
    coord_mag = np.ceil(np.log(np.max(
        np.concatenate((startpt,endpt,mesh),axis=None))))
    EPS = 10**(coord_mag-7)

    # Get full travel vector
    vec = endpt-startpt

    DIM = len(startpt)

    if DIM == 2:
        # intersection comes from _geom.seg_intersect_2D
        x = intersection[0]    # (x,y) coordinates of intersection
        t_I = intersection[1]  # btwn 0 & 1, fraction of movement traveled so far
        Q0 = intersection[2]   # edge of mesh element intersected
        Q1 = intersection[3]   # edge of mesh element intersected
        idx = intersection[4]  # index into mesh that will yield Q0 and Q1

        # Get mesh element vector
        Qvec = Q1-Q0
        Qvec_u = Qvec/np.linalg.norm(Qvec)

        # Find direction indictor for which side of the mesh element the 
        #   agent hit the element.
        agent_prev_loc = startpt + vec*(t_I-0.00001)
        dir_vec = agent_prev_loc-x
        Q_orth_u = np.array([Qvec_u[1],-Qvec_u[0]])
        # side_signum will orient back in the direction the agent came from
        side_signum = np.dot(dir_vec,Q_orth_u)\
            /np.linalg.norm(np.dot(dir_vec,Q_orth_u))
        # Get a normal to the mesh element that points back toward where the 
        #   agent came from.
        Q_norm_u = side_signum*Q_orth_u
        
        # get a perpendicular INTO the element
        # Qperp_in = lambda t: -side_signum*np.array([Qvec(t)[1],-Qvec(t)[0]])

        # Vector projection of vec onto direction Q is obtained via 
        #   Q/||Q||*dot(vec,Q/||Q||) = Q*dot(vec,Q)/||Q||**2.
        proj_vec = Qvec*np.dot(vec,Qvec)/np.dot(Qvec,Qvec)
        proj_vec_u = proj_vec/np.linalg.norm(proj_vec)

    elif DIM == 3:
        # intersection comes from _geom.seg_intersect_3D_triangles
        x = intersection[0]     # (x,y,z) coordinates of intersection
        t_I = intersection[1]   # btwn 0 & 1, fraction of movement traveled so far
        Q0 = intersection[2]    # first vertex of mesh element intersected
        Q1 = intersection[3]    # second vertex of mesh element intersected
        Q2 = intersection[4]    # third vertex of mesh element intersected
        idx = intersection[5]   # index into mesh that will yield Q0,Q1,Q2

        # Get mesh element vectors
        Qvec10 = Q1-Q0
        Qvec21 = Q2-Q1

        # Find direction indictor for which side of the mesh element the 
        #   agent hit the element.
        agent_prev_loc = startpt + vec*(t_I-0.00001)
        dir_vec = agent_prev_loc-x
        Q_norm = np.cross(Qvec10, Qvec21)
        # side_signum will orient back in the direction the agent came from
        side_signum = np.dot(dir_vec,Q_norm)\
            /np.linalg.norm(np.dot(dir_vec,Q_norm))
        # Make the normal to the mesh element point back toward where the 
        #   agent came from.
        Q_norm = side_signum*Q_norm
        Q_norm_u = Q_norm/np.linalg.norm(Q_norm)

        # Vector projection onto mesh element is obtained by subtracting the
        #   projection onto Q_norm.
        # Vector projection onto Q_norm is 
        #   Qn/||Qn||*dot(vec,Qn/||Qn||) = Qn*dot(vec,Qn)/||Qn||**2
        proj_vec = (vec - Q_norm*np.dot(vec,Q_norm)/np.dot(Q_norm,Q_norm))
        proj_vec_u = proj_vec/np.linalg.norm(proj_vec)

    # Position of agent at time t
    proj_to_pt = lambda t: (t-t_I)*proj_vec + x
    
    # Projected position at end of time period
    slide_pt = proj_to_pt(1)

    if 1-t_I < 10e-7:
        # Special case where we are practically finished with this time step.
        # Perturb away from the mesh element and return.
        return x + EPS*Q_norm_u
        
    ##########                                             ##########
    #####             Test for sliding off the end              #####
    ##########                                             ##########

    if DIM == 2:
        mesh_el_end_len = np.linalg.norm(Qvec)
        Q0_crit_dist = np.linalg.norm(slide_pt - Q0)
        Q1_crit_dist = np.linalg.norm(slide_pt - Q1)
        # Since we are sliding on the mesh element, if the distance from
        #   our new location to either of the mesh endpoints is greater
        #   than the length of the mesh element, we must have gone beyond
        #   the segment somewhere in the past.

        # Check to see if and when we went past the end of the mesh element.
        # Solve (t_edge-t_I)*Qvec*np.dot(vec,Qvec)/np.dot(Qvec,Qvec)+x=Q0 or Q1
        if Q1_crit_dist > mesh_el_end_len+EPS and Q1_crit_dist > Q0_crit_dist:
            # went past Q0
            t_edge = np.linalg.norm((Q0-x)*np.dot(Qvec,Qvec)/np.dot(vec,Qvec)+t_I*Qvec)\
                        /np.linalg.norm(Qvec)
            Q_edge = Q0
        elif Q0_crit_dist > mesh_el_end_len+EPS and Q0_crit_dist > Q1_crit_dist:
            t_edge = np.linalg.norm((Q1-x)*np.dot(Qvec,Qvec)/np.dot(vec,Qvec)+t_I*Qvec)\
                        /np.linalg.norm(Qvec)
            Q_edge = Q1
        else:
            t_edge = None
    elif DIM == 3:
        # Detect sliding off 2D edge using seg_intersect_2D
        # Construct lists of first and second points for the line segments
        Q0_list = np.array(intersection[2:5]) # Q0,Q1,Q2
        Q1_list = Q0_list[(1,2,0),:] # Q1,Q2,Q0
        tri_intersect = _geom.seg_intersect_2D(x, slide_pt, Q0_list, Q1_list)
        if tri_intersect is None:
            t_edge = None
        else:
            # Get time of intersection
            t_edge = t_I + (1-t_I)*tri_intersect[1]
            # Get point of intersection
            x_edge = tri_intersect[0]
            # Get side of intersection
            idx_edge = tri_intersect[4]
            
    ##########                                                  ##########
    #####       Algorithms for going past end of mesh element        #####
    ##########                                                  ##########

    if t_edge is not None:
        # Find any adjacent mesh segments on the end we went past that would
        #   cause intersection
        if DIM == 2:
            # In 2D, this is any mesh element that contains the endpoint 
            #   we went past except the current mesh element
            if Q1_crit_dist > Q0_crit_dist:
                # went past Q0
                pt_bool = np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q0, axis=1), 0)
            else:
                # went past Q1
                pt_bool = np.isclose(np.linalg.norm(mesh.reshape(
                    (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q1, axis=1), 0)
            pt_bool = pt_bool.reshape((mesh.shape[0],mesh.shape[1]))
            # remove current mesh element
            pt_bool[idx,:] = False
            adj_mesh = mesh[np.any(pt_bool,axis=1)]
            
            # Determine if there are intersections with adjacent mesh elements
            if len(adj_mesh) > 0:
                # index into mesh for adjacent mesh elements
                adj_mesh_idx = np.any(pt_bool,axis=1).nonzero()[0]
                # get vectors in adjacent meshes oriented away from current edgepoint
                pt_bool_0 = pt_bool[adj_mesh_idx,0]
                adj_vec = np.zeros((adj_mesh.shape[0],adj_mesh.shape[2]))
                # if first entry is the focal edgepoint, get vector from that
                adj_vec[pt_bool_0] = adj_mesh[pt_bool_0,1,:] - \
                                        adj_mesh[pt_bool_0,0,:]
                # otherwise, get vector from the second edgepoint as the focal one
                adj_vec[~pt_bool_0] = adj_mesh[~pt_bool_0,0,:] - \
                                        adj_mesh[~pt_bool_0,1,:]
                # intersection cases will have an angle of -pi/2 to pi/2 between
                #   Q_norm_u and the adjacent mesh element oriented from the 
                #   edge the agent is on. That means the dot product is positive.
                intersect_bool = np.dot(adj_vec, Q_norm_u) >= 0
            else:
                intersect_bool = np.array([False])

        elif DIM == 3:
            if idx_edge == 0:
                # Went past Q0Q1
                pt_bool0 = np.isclose(np.linalg.norm(mesh.reshape(
                        (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q0, axis=1), 0)
                pt_bool1 = np.isclose(np.linalg.norm(mesh.reshape(
                        (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q1, axis=1), 0)
            elif idx_edge == 1:
                # Went past Q1Q2
                pt_bool0 = np.isclose(np.linalg.norm(mesh.reshape(
                        (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q1, axis=1), 0)
                pt_bool1 = np.isclose(np.linalg.norm(mesh.reshape(
                        (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q2, axis=1), 0)
            elif idx_edge == 2:
                # Went past Q2Q0
                pt_bool0 = np.isclose(np.linalg.norm(mesh.reshape(
                        (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q2, axis=1), 0)
                pt_bool1 = np.isclose(np.linalg.norm(mesh.reshape(
                        (mesh.shape[0]*mesh.shape[1],mesh.shape[2]))-Q0, axis=1), 0)
            pt_bool0 = pt_bool0.reshape((mesh.shape[0],mesh.shape[1]))
            pt_bool1 = pt_bool1.reshape((mesh.shape[0],mesh.shape[1]))
            # remove current mesh element
            pt_bool0[idx,:] = False
            pt_bool1[idx,:] = False
            adj_mesh = mesh[np.logical_and(np.any(pt_bool0,axis=1),
                                        np.any(pt_bool1,axis=1))]
            
            # Determine if there are intersections with adjacent mesh elements
            if len(adj_mesh) > 0:
                # index into mesh for adjacent mesh elements
                adj_mesh_idx = np.logical_and(np.any(pt_bool0,axis=1),
                                            np.any(pt_bool1,axis=1)).nonzero()[0]
                # we need the mesh point of the adjacent mesh that is not part of the
                #   edge of intersection
                other_bool = np.logical_not(np.logical_or(pt_bool0[adj_mesh_idx,:],
                                                        pt_bool1[adj_mesh_idx,:]))
                adj_Q_other = adj_mesh[other_bool,:]
                # get vector pointed away from shared edge on adjacent elements
                adj_vec = adj_Q_other - x_edge
                # intersection cases will have an angle of -pi/2 to pi/2 between
                #   Q_norm_u and this vector oriented away from the edge the agent 
                #   is on. That means the dot product is positive.
                intersect_bool = np.dot(adj_vec, Q_norm_u) >= 0
            else:
                intersect_bool = np.array([False])

        # Handle any intersections with adjacent elements and return
        if np.any(intersect_bool):
            # Get info about the relevant adjacent elements
            adj_vec = adj_vec[intersect_bool]
            adj_vec_u = adj_vec/np.linalg.norm(adj_vec, axis=-1)
            adj_vec_idx = adj_mesh_idx[intersect_bool]
            if adj_vec.shape[0] > 1:
                # get the mesh element that forms the most acute angle with
                #   the current mesh element. This is equivalent to the largest
                #   angle between proj_vec and adj_vec
                # clip protects against roundoff error
                proj_adj_angles = np.arccos(
                    np.clip(np.dot(proj_vec_u,adj_vec_u),-1.0, 1.0)
                    ) # all within interval [0,pi]
                adj_vec_int_idx = np.argmax(proj_adj_angles)
                adj_vec = adj_vec[adj_vec_int_idx,:]
                adj_vec_u = adj_vec_u[adj_vec_int_idx,:]
                adj_idx = adj_vec_idx[adj_vec_int_idx]
                proj_adj_angle = proj_adj_angles[adj_vec_int_idx]
            else:
                adj_vec = adj_vec[0,:]
                adj_vec_u = adj_vec_u[0,:]
                adj_idx = adj_mesh_idx[intersect_bool][0]
                proj_adj_angle = np.arccos(
                    np.clip(np.dot(proj_vec_u,adj_vec_u),-1.0, 1.0))
                
            # Treat case of sliding back to a previous mesh element and the
            #   case of a sharp angle
            if (prev_idx is not None and prev_idx == adj_idx) or\
                proj_adj_angle >= np.pi/2:
                # Back away from the intersection point slightly in the 
                #   direction that bisects the angle between the mesh 
                #   elements for stay put.
                mid_vec = (adj_vec_u - proj_vec_u)*0.5
                if DIM == 2:
                    return Q_edge + EPS*mid_vec
                elif DIM == 3:
                    return x_edge + EPS*mid_vec
            # Otherwise, slide on adjacent mesh element.
            else:
                # Repeat project_and_slide on new segment.
                if DIM == 2:
                    adj_intersect = (Q_edge, t_edge, 
                                        mesh[adj_idx,0,:],
                                        mesh[adj_idx,1,:], adj_idx)
                    # Supply new start/end pts based on new intersection point 
                    #   and original trajectory
                    newstartpt = Q_edge - t_edge*vec
                    newendpt = Q_edge + (1-t_edge)*vec
                elif DIM == 3:
                    adj_intersect = (x_edge, t_edge, mesh[adj_idx,0,:],
                                    mesh[adj_idx,1,:], mesh[adj_idx,2,:],
                                    adj_idx)
                    # Supply new start/end pts based on new intersection point 
                    #   and original trajectory
                    newstartpt = x_edge - t_edge*vec
                    newendpt = x_edge + (1-t_edge)*vec

                return _project_and_slide_static(newstartpt, newendpt, 
                                                        adj_intersect, 
                                                        mesh, max_meshpt_dist, 
                                                        prev_idx=idx)
                
    ##########                                         ##########
    #####       No intersection with adjacent element       #####
    ##########                                         ##########

    if t_edge is not None:
        # Continue on the original trajectory from the time of separation
        #   from the mesh element.
        # Recursively check for more intersections.
        if DIM == 2:
            newstartpt = Q_edge + EPS*Q_norm_u
        elif DIM == 3:
            newstartpt = x_edge + EPS*Q_norm_u
        newendpt = newstartpt + (1-t_edge)*vec
        # recursion on prev. eligible mesh elements and treating t_edge 
        #   as the start time. Only look for intersections with subset
        #   of full eligible mesh.
        close_mesh = _get_eligible_static_mesh_elements(newstartpt, 
                                                                newendpt, mesh, 
                                                                max_meshpt_dist*2/3)
        if DIM == 2:
            intersection_n = _geom.seg_intersect_2D(newstartpt, newendpt,
                                                    close_mesh[:,0,:], 
                                                    close_mesh[:,1,:])
        elif DIM == 3:
            intersection_n = _geom.seg_intersect_3D_triangles(newstartpt, newendpt,
                            close_mesh[:,0,:], close_mesh[:,1,:], close_mesh[:,2,:])
        if intersection_n is None:
            return newendpt
        else:
            # Get idx in intersection_n to match full mesh instead of close_mesh
            elem = close_mesh[intersection_n[-1]]
            idx_n = np.argwhere((elem == mesh).all(axis=(1,2)))[0,0]
            intersection_n = (*intersection_n[:-1], idx_n)
            new_loc = _project_and_slide_static(newstartpt, newendpt,
                                                        intersection_n, mesh, 
                                                        max_meshpt_dist)
        return new_loc
    else:
        # Ended on mesh element
        # Perturb back toward where the agent came from.
        return slide_pt + EPS*Q_norm_u



def _project_and_slide_moving(startpt, endpt, intersection, mesh_start, 
                                mesh_end, max_meshpt_dist, max_mov, 
                                prev_idx=None):
    '''Once we have an intersection point with an immersed mesh, slide the 
    agent along the moving mesh for its remaining movement (frictionless 
    boundary interaction), and then determine what happens if we fall off 
    the edge of the element or if the element rotates away.
    
    NOTE: Because we do not know the mass of the agents or the properties
    of the fluid, we neglect inertial and drag forces in this computation.

    Parameters
    ----------
    startpt : length 2 or 3 array
        original start point of movement, before intersection
    endpt : length 2 or 3 array
        original end point of movement, w/o intersection
    intersection : list-like of data
        result of seg_intersect_2D_multilinear_poly. various information 
        about the intersection with the immersed mesh element
    mesh_start : Nx2x2 or Nx3x3 array 
        eligible (nearby) mesh elements for interaction as they are at the 
        start time
    mesh_end : Nx2x2 or Nx3x3 array 
        eligible (nearby) mesh elements for interaction as they are at the 
        end time
    max_meshpt_dist : float
        max distance between two points on a mesh element (used to determine 
        how far away from startpt to search for mesh elements). 
        Passed-through here solely in case of recursion with 
        _apply_internal_moving_BC.
    max_mov : float
        maximum distance any mesh vertex moved. Passed-through here solely 
        in case of recursion with _apply_internal_moving_BC.
    prev_idx : int, optional
        in recursion with adjoining mesh elements, this prevents an infinite 
        recusion with, e.g., mesh elements in an acute angle.

    Returns
    -------
    newendpt : length 2 array
        new endpoint for movement after projection
    '''

    # small number to perturb off of the actual boundary in order to avoid
    #   roundoff errors that would allow penetration
    # base its magnitude off of the given coordinate points
    coord_mag = np.ceil(np.log(np.max(
        np.concatenate((startpt,endpt,mesh_start,mesh_end),axis=None))))
    EPS = 10**(coord_mag-7)

    DIM = len(startpt)

    if DIM != 2:
        raise NotImplementedError("3D moving meshes not supported for sliding.")
    
    x = intersection[0]    # (x,y) coordinates of intersection
    t_I = intersection[1]  # btwn 0 & 1, fraction of movement traveled so far
    Q0 = intersection[2]   # edge of mesh element at time of intersection
    Q1 = intersection[3]   # edge of mesh element at time of intersection
    idx = intersection[4]  # index into mesh_start/end above
    
    # Get full travel vector, to be integrated from t_I to 1
    vec = endpt-startpt

    Q_tI = Q1-Q0 # vector in direction of mesh elem at t_I
    Q_end = mesh_end[idx,1,:] - mesh_end[idx,0,:] # same but at t=1
    Qvec = lambda t: ((1-t)*Q_tI+(t-t_I)*Q_end)/(1-t_I) # same but as func of t
    Q0_t = lambda t: ((1-t)*Q0+(t-t_I)*mesh_end[idx,0,:])/(1-t_I) # interp of Q0

    # Find direction indicator for which side of the mesh element the agent 
    #   hit the element. Base this off of location of mesh and agent a small
    #   time before intersection.
    agent_prev_loc = startpt + vec*(t_I-0.00001)
    # Barycentric position of intersection on mesh element from Q0
    s_I = (x.sum()-Q0.sum())/Q_tI.sum()
    x_prev_loc = s_I*Qvec(t_I-0.00001) + Q0_t(t_I-0.00001)
    dir_vec = agent_prev_loc-x_prev_loc
    Q_tI_orth = np.array([Q_tI[1],-Q_tI[0]])
    # side_signum will orient back in the direction the agent came from
    side_signum = np.dot(dir_vec,Q_tI_orth)\
        /np.linalg.norm(np.dot(dir_vec,Q_tI_orth))

    # get a perpendicular into the element
    Qperp = lambda t: -side_signum*np.array([Qvec(t)[1],-Qvec(t)[0]])
    # derivatives
    Qvec_d = (Q_end-Q_tI)/(1-t_I)
    Q0_d = (mesh_end[idx,0,:]-Q0)/(1-t_I)

    # Vector projection of vec onto direction Q is obtained via 
    #   Q/||Q||*dot(vec,Q/||Q||) = Q*dot(vec,Q)/||Q||**2.
    # Agent movement is the sum of two components:
    #   1) Movement along Qvec(t) due to projection of agent velocity (mesh 
    #       element is frictionless so does not impact this)
    #   2) Movement ortho to Qvec(t) due to movement of the mesh element
    # This means that dx/dt is the projection of vec onto direction Qvec plus 
    #   dx/dt due to mesh element movement projected onto direction Qperp.
    # dx/dt due to mesh element is given by 
    #   dx/dt = dQ0/dt + dQvec/dt*s(t) + Qvec*ds/dt where s(T) is the 
    #   Berycentric coordinate of x(t) on the mesh element,
    #   s(t) = ||x(t)-Q0(t)||/||Qvec(t)||. The third term in dx/dt can be 
    #   neglected because the dot product of Qvec and Qperp is zero.

    # Derivative of agent movement
    def x_DE(t,x):
        return Qvec(t)*np.dot(vec,Qvec(t))/np.dot(Qvec(t),Qvec(t)) +\
        Qperp(t)*np.dot(Q0_d + Qvec_d*np.linalg.norm(x-Q0_t(t))/np.linalg.norm(Qvec(t)),
                        Qperp(t))/np.dot(Qperp(t),Qperp(t))
    
    # Solve for position at time t
    proj_to_pt = lambda t: integrate.solve_ivp(x_DE, t_span=(t_I,t), y0=x, 
                                                rtol=1e-7, atol=1e-12).y[:,-1]

    # Derivative as a function of t alone for least squares optimization
    proj_prime = lambda t: x_DE(t, proj_to_pt(t))
    
    # integrate to 1 to determine the final sliding location on the mesh element
    slide_pt = proj_to_pt(1)

    if 1-t_I < 10e-7:
        # Special case where we are practically finished with this time step.
        # Get a normal to the final position of mesh element that points 
        #   back toward where the agent came from. perturb in that direction
        norm_out_u = side_signum*np.array([Q_end[1],-Q_end[0]])
        norm_out_u /= np.linalg.norm(norm_out_u)
        # pull slide_pt back a bit for numerical stability and return
        return x + EPS*norm_out_u
    
    ##########                                             ##########
    #####             Test for sliding off the end              #####
    ##########                                             ##########

    mesh_el_end_len = np.linalg.norm(Q_end)
    Q0_crit_dist = np.linalg.norm(slide_pt - mesh_end[idx,0,:])
    Q1_crit_dist = np.linalg.norm(slide_pt - mesh_end[idx,1,:])
    # Since we are sliding on the mesh element, if the distance from
    #   our new location to either of the mesh endpoints is greater
    #   than the length of the mesh element, we must have gone beyond
    #   the segment somewhere in the past.
    went_past_el_bool = (Q0_crit_dist > mesh_el_end_len+EPS) or ( 
                            Q1_crit_dist > mesh_el_end_len+EPS)
    
    # Because mesh elements are linearly interpolated between start and end 
    #   states, they stretch or contract quadratically. Also, the direction 
    #   of travel on the mesh element can change at most once. If we haven't
    #   detected a trip past one of the endpoints already, establish the 
    #   existence and feasibility of these critical points and check them in 
    #   addition to the state at the end of the timestep.

    if not went_past_el_bool:
        t_crit_elem_denom = np.dot(Q_tI+Q_end,np.ones(2))-2*np.dot(Q_tI,Q_end)
        if t_crit_elem_denom != 0:
            t_crit_elem = np.dot(Q_tI,1-Q_end)/t_crit_elem_denom
        else:
            t_crit_elem = -1
        if t_I < t_crit_elem < 1:
            t_crit = t_crit_elem
        else:
            t_crit = None
        
        t_crit_x_denom = np.dot(Q_tI-Q_end,vec)
        if t_crit_x_denom != 0:
            t_crit_x = np.dot(Q_tI,vec)/t_crit_x_denom
        else:
            t_crit_x = -1
        if t_I < t_crit_x < 1:
            if t_crit is None or t_crit > t_crit_x:
                t_crit = t_crit_x

        if t_crit is not None:
            slide_pt_crit = proj_to_pt(t_crit)
            mesh_el_crit_len = np.linalg.norm(Qvec(t_crit))
            Q0_crit_dist = np.linalg.norm(slide_pt_crit - Q0_t(t_crit))
            Q1_crit_dist = np.linalg.norm(slide_pt_crit - (Q0_t(t_crit)+Qvec(t_crit)))
            went_past_el_bool = (Q0_crit_dist > mesh_el_crit_len+EPS) or ( 
                                    Q1_crit_dist > mesh_el_crit_len+EPS)

    if went_past_el_bool:
        # check to see when we went past the end of the mesh element.
        # This is a non-linear least squares minimization problem
        if Q1_crit_dist > Q0_crit_dist:
            # went past Q0
            # location of Q0 as a function of time, plus its derivative
            Q_edge = lambda t: ((1-t)*Q0 + (t-t_I)*mesh_end[idx,0,:])/(1-t_I)
            Q_edge_prime = (mesh_end[idx,0,:]-Q0)/(1-t_I)
        else:
            # went past Q1
            # location of Q1 as a function of time, plus its derivative
            Q_edge = lambda t: ((1-t)*Q1 + (t-t_I)*mesh_end[idx,1,:])/(1-t_I)
            Q_edge_prime = (mesh_end[idx,1,:]-Q1)/(1-t_I)
        # function that gives distance between projected location of 
        #   agent at time t and the relevant edge of the mesh element.
        #   Find roots of the square distance.
        resid = lambda t: proj_to_pt(t[0])-Q_edge(t[0])
        # derivative
        jac = lambda t: np.array([proj_prime(t[0])-Q_edge_prime]).T
        # Solve the non-linear least squares problem for a root local to t_I.
        sol = optimize.least_squares(resid, x0=t_I, jac=jac, bounds=(t_I,1),
                                        method='dogbox', ftol=None, gtol=None)
        # check solution
        if not sol.success:
            raise RuntimeError("LSQ did not converge.")
        t_edge = sol.x[0]
        # for debugging
        # print(f't_edge = {t_edge}')
    
    ##########                                             ##########
    #####                Test for rotating away                 #####
    ##########                                             ##########

    # Check for the mesh element rotating until it is moving faster 
    #   than vec in the direction orth to the element (before reaching the 
    #   edge of the element). The speed of Qperp should be a convex up 
    #   function of position x along the element, which means we can check 
    #   the final time to see if this is possible.
    if went_past_el_bool:
        t_end = t_edge
    else:
        t_end = 1

    vec_Q_perp = np.dot(vec,Qperp(t_end))/np.dot(Qperp(t_end),Qperp(t_end))
    x_Q_perp = np.dot(Q0_d + Qvec_d*np.linalg.norm(proj_to_pt(t_end)-Q0_t(t_end))
                    /np.linalg.norm(Qvec(t_end)),
                    Qperp(t_end))/np.dot(Qperp(t_end),Qperp(t_end))
    rotated_past_bool = x_Q_perp > vec_Q_perp
    
    if rotated_past_bool:
        # Determine when the velocity magnitudes in the ortho direction 
        #   matched. This is a root finding problem.
        spd_diff = lambda t: np.dot(Q0_d + 
            Qvec_d*np.linalg.norm(proj_to_pt(t)-Q0_t(t))/np.linalg.norm(Qvec(t)),
            Qperp(t))/np.dot(Qperp(t),Qperp(t)) -\
            np.dot(vec,Qperp(t))/np.dot(Qperp(t),Qperp(t))
        
        sol = optimize.root_scalar(spd_diff, method='brentq', bracket=(t_I,t_end))

        if not sol.converged:
            raise RuntimeError("Brenq did not converge.")
        t_rot = sol.root
        # for debugging
        # print(f't_rot = {t_rot}')
        # rotated away before end of element was reached
        went_past_el_bool = False

    ##########                                                  ##########
    #####       Algorithms for going past end of mesh element        #####
    ##########                                                  ##########
    # If we went past the end of the mesh element, detect intersection with 
    #   adjoining elements.
    if went_past_el_bool:
        # get a unit vector in the direction of travel at edge time
        Qvec_tedge = Qvec(t_edge)
        Qslide_vec_u = Qvec_tedge*np.dot(vec,Qvec_tedge)
        Qslide_vec_u /= np.linalg.norm(Qslide_vec_u)
        # projection of vec onto mesh element in dir of travel at edge time
        # proj_Q = np.dot(vec,Qslide_vec_u)*Qslide_vec_u
        # unit normal to current mesh element toward where the agent came from
        norm_out_u = side_signum*np.array([Qvec_tedge[1],-Qvec_tedge[0]])
        norm_out_u /= np.linalg.norm(norm_out_u)

        # Now, find adjacent mesh segments on the end we went past. 
        #   This is any mesh element that contains the endpoint we went past
        #   except the current mesh element
        if Q1_crit_dist > Q0_crit_dist:
            # went past Q0
            pt_bool = np.isclose(np.linalg.norm(mesh_end.reshape(
                (mesh_end.shape[0]*mesh_end.shape[1],mesh_end.shape[2]))
                -mesh_end[idx,0,:], axis=1), 0)
        else:
            # went past Q1
            pt_bool = np.isclose(np.linalg.norm(mesh_end.reshape(
                (mesh_end.shape[0]*mesh_end.shape[1],mesh_end.shape[2]))
                -mesh_end[idx,1,:], axis=1), 0)
        pt_bool = pt_bool.reshape((mesh_end.shape[0],mesh_end.shape[1]))
        # remove current mesh element
        pt_bool[idx,:] = False
        adj_mesh_end = mesh_end[np.any(pt_bool,axis=1)]

        # if there are any adjacent mesh elements, get their location at 
        #   time t_edge as well and calculate the angle at that time between 
        #   the current mesh element and the adjacent ones. Then react 
        #   accordingly.
        ################
        if len(adj_mesh_end) > 0:
            # index into mesh_end for adjacent mesh elements
            adj_mesh_end_idx = np.any(pt_bool,axis=1).nonzero()[0]
            # location of adj mesh at time t_edge
            adj_mesh_newstart = mesh_start[adj_mesh_end_idx]*(1-t_edge)\
                                +adj_mesh_end*t_edge
            # get vectors in adjacent meshes oriented away from current edgepoint
            pt_bool_0 = pt_bool[adj_mesh_end_idx,0]
            adj_vec = np.zeros((adj_mesh_newstart.shape[0],adj_mesh_newstart.shape[2]))
            # if first entry is the focal edgepoint, get vector from that
            adj_vec[pt_bool_0] = adj_mesh_newstart[pt_bool_0,1,:] - \
                                    adj_mesh_newstart[pt_bool_0,0,:]
            # otherwise, get vector from the second edgepoint as the focal one
            adj_vec[~pt_bool_0] = adj_mesh_newstart[~pt_bool_0,0,:] - \
                                    adj_mesh_newstart[~pt_bool_0,1,:]
            # intersection cases will have an angle of -pi/2 to pi/2 between
            #   norm_out_u and the adjacent mesh element oriented from the 
            #   edge the agent is on. That means the dot product is positive.
            intersect_bool = np.dot(adj_vec, norm_out_u) >= 0
        else:
            intersect_bool = np.array([False])
        ################

        if np.any(intersect_bool):
            ########  Went past and intersected adjoining element! ########
            # Get info about it
            adj_vec = adj_vec[intersect_bool]
            adj_vec_u = adj_vec/np.linalg.norm(adj_vec, axis=-1)
            adj_vec_idx = adj_mesh_end_idx[intersect_bool]
            adj_mesh_newstart = adj_mesh_newstart[intersect_bool]
            if adj_vec.shape[0] > 1:
                # get the one that is most acute on the side of norm_out_u
                # This is equivalent to the largest angle between Qslide_vec_u 
                #   and adj_vec_u
                # clip protects against roundoff error
                proj_adj_angles = np.arccos(
                    np.clip(np.dot(Qslide_vec_u, adj_vec_u),-1.0, 1.0)
                    ) # all within interval [0,pi]
                adj_vec_int_idx = np.argmax(proj_adj_angles)
                adj_vec = adj_vec[adj_vec_int_idx,:]
                adj_vec_u = adj_vec_u[adj_vec_int_idx,:]
                adj_idx = adj_vec_idx[adj_vec_int_idx]
                proj_adj_angle = proj_adj_angles[adj_vec_int_idx]
            else:
                adj_vec = adj_vec[0,:]
                adj_vec_u = adj_vec_u[0,:]
                adj_idx = adj_mesh_end_idx[intersect_bool][0]
                adj_vec_int_idx = 0
                proj_adj_angle = np.arccos(
                        np.clip(np.dot(Qslide_vec_u, adj_vec_u),-1.0, 1.0))
            # NOTE: This intersection happens at the same time as sliding 
            #   off of the last element because they are joined together.

            # Treat case of sliding back to a previous mesh element and the
            #   case of a sharp angle
            if (prev_idx is not None and prev_idx == adj_idx) or\
                proj_adj_angle >= np.pi/2:
                # Back away from the intersection point slightly in the 
                #   direction that bisects the angle between the mesh 
                #   elements for stay put.
                if pt_bool[adj_idx,0]: # first entry is focal edgepoint
                    adj_vec_end = mesh_end[adj_idx,1,:] - mesh_end[adj_idx,0,:]
                else:
                    adj_vec_end = mesh_end[adj_idx,0,:] - mesh_end[adj_idx,1,:]
                adj_vec_end_u = adj_vec_end/np.linalg.norm(adj_vec_end)
                Q_end_u = Q_end/np.linalg.norm(Q_end)
                if Q1_crit_dist < Q0_crit_dist:
                    # went past Q1, not Q0
                    Q_end_u *= -1
                mid_vec = (adj_vec_end_u + Q_end_u)*0.5
                return Q_edge(1) + EPS*mid_vec
            else:
                # Repeat project_and_slide_moving on new segment.
                adj_intersect = (Q_edge(t_edge), t_edge, 
                                    adj_mesh_newstart[adj_vec_int_idx,0,:],
                                    adj_mesh_newstart[adj_vec_int_idx,1,:], adj_idx)
                # Supply new start/end pts based on new intersection point 
                #   and original trajectory
                newstartpt = adj_intersect[0] - t_edge*vec
                newendpt = adj_intersect[0] + (1-t_edge)*vec

                # for debugging
                # print(f'newstartpt = {list(newstartpt)}')
                # print(f'newendpt = {list(newendpt)}')
                return _project_and_slide_moving(newstartpt, newendpt, 
                                                adj_intersect, 
                                                mesh_start, mesh_end, 
                                                max_meshpt_dist, max_mov,
                                                prev_idx=idx)      

    ##########                                                  ##########
    #####   Only reached if we did not intersect adjacent element    #####
    ##########                                                  ##########
    # If the mesh element rotated out of the way of the agent's original 
    #   trajectory or we did not intersect an adjoining mesh element at the 
    #   edge of the previous one, continue on the original trajectory from 
    #   the time of separation. Recursively check for any more intersections
    #   in those cases.
    if rotated_past_bool:
        # add EPS for separation from element
        norm_out_u = side_signum*np.array([Qvec(t_rot)[1],-Qvec(t_rot)[0]])
        norm_out_u /= np.linalg.norm(norm_out_u)
        newstartpt = proj_to_pt(t_rot) + EPS*norm_out_u
        newendpt = newstartpt + (1-t_rot)*vec
        mesh_now = mesh_start*(1-t_rot) + mesh_end*t_rot
        # for debugging
        # print(f'newstartpt = {list(newstartpt)}')
        # print(f'newendpt = {list(newendpt)}')
    elif went_past_el_bool:
        # Slid off the end and encountered nothing.
        newstartpt = Q_edge(t_edge) + EPS*norm_out_u
        newendpt = newstartpt + (1-t_edge)*vec
        mesh_now = mesh_start*(1-t_edge) + mesh_end*t_edge
        # for debugging
        # print(f'newstartpt = {list(newstartpt)}')
        # print(f'newendpt = {list(newendpt)}')
    else:
        ######### Ended on mesh element ##########
        # get a normal to the final position of mesh element that points 
        #   back toward where the agent came from. perturb in that direction
        norm_out_u = side_signum*np.array([Q_end[1],-Q_end[0]])
        norm_out_u /= np.linalg.norm(norm_out_u)
        return slide_pt + EPS*norm_out_u

    # recursion on prev. eligible mesh elements and treating t_rot or t_edge 
    #   as the start time. Only look for intersections with subset of full 
    #   eligible mesh.
    close_mesh_start, close_mesh_end = \
        _get_eligible_moving_mesh_elements(newstartpt, newendpt, 
                                                    mesh_now, mesh_end, max_mov, 
                                                    max_meshpt_dist*2/3)
    intersection_n = _geom.seg_intersect_2D_multilinear_poly(newstartpt, newendpt,
                            close_mesh_start[:,0,:], close_mesh_start[:,1,:],
                            close_mesh_end[:,0,:], close_mesh_end[:,1,:])
    if intersection_n is None:
        return newendpt
    else:
        # Get idx in intersection_n to match full mesh instead of close_mesh
        elem = close_mesh_start[intersection_n[4]]
        idx_n = np.argwhere((elem == mesh_now).all(axis=(1,2)))[0,0]
        intersection_n = (*intersection_n[:-1], idx_n)
        new_loc = _project_and_slide_moving(newstartpt, newendpt,
                                                    intersection_n, mesh_now,
                                                    mesh_end, max_meshpt_dist,
                                                    max_mov)
    return new_loc
