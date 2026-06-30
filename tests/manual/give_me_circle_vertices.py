'''
Create a 2D circle of vertices for testing purposes
'''

import numpy as np

def  give_Me_Immersed_Boundary_Geometry(ds,r,center):
    '''
    Arguments:
        ds: Lagrangian spacing
        r: radius of circle
        center: [x,y] center of circle
    '''

    dtheta = np.arccos(1 - ds**2/(2*r**2))
    theta_list = np.arange(0,2*np.pi,dtheta)

    xLag = center[0] + r*np.cos(theta_list)
    yLag = center[1] + r*np.sin(theta_list)

    N = len(xLag)

    with open('circle.vertex','w') as fobj:
        fobj.write('{}\n'.format(N))
        for n in range(N):
            fobj.write('{:1.16e} {:1.16e}\n'.format(xLag[n], yLag[n]))