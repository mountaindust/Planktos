"""
Test suite for planktos: Environment and Swarm classes.
For use with py.test package.

Created on April 04 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
"""

import pytest
import numpy as np
import numpy.ma as ma
import planktos
from planktos import motion

############                    Decorators                ############

# @pytest.mark.slow : slow test, run only with --runslow
# @pytest.mark.vkt : won't run if unable to load vtk

############   Basic Overrides to test different physics  ############

class highRe_massive_swarm(planktos.Swarm):

    def apply_agent_model(self, dt):
        '''Uses projectile motion'''

        # Get drift/drag/inertia due to fluid using high Reynolds number equation
        #   of motion.
        ode = motion.highRe_massive_drift(self)

        # mu property of Swarm will automatically be added to ode equations
        #   in the right way.
        return motion.Euler_brownian_motion(self, dt, ode=ode)



class lowRe_massive_swarm(planktos.Swarm):

    def apply_agent_model(self, dt):
        '''Uses Haller and Sapsis'''

        # Get drift/drag/inertia due to fluid using high Reynolds number equation
        #   of motion.
        ode = motion.inertial_particles(self)

        # mu property of Swarm will automatically be added to ode equations
        #   in the right way.
        return motion.Euler_brownian_motion(self, dt, ode=ode)



###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_basic():
    '''Test no-flow, basic stuff'''
    envir = planktos.Environment()
    sw = envir.add_swarm()
    for ii in range(10):
        sw.move(0.25)
    assert envir.time == 2.5

    sw = planktos.Swarm()
    for ii in range(10):
        sw.move(0.25)
    assert sw.envir.time == 2.5

    envir2 = planktos.Environment()
    sw = planktos.Swarm()
    envir2.add_swarm(sw)
    assert envir2.swarms[0] is sw, "pre-existing Swarm not added"
    for ii in range(10):
        sw.move(0.25)



def test_indiv_variation():
    '''With no-flow, test individual variation using dataframe props.
    This mainly just checks that there are no syntax errors.'''
    swarm_size = 30
    # 2D
    envir = planktos.Environment()
    sw = envir.add_swarm(swarm_size=swarm_size)
    sw.add_prop('mu', [np.random.randn(2) for ii in range(swarm_size)])
    assert 'mu' not in sw.shared_props
    assert sw.get_prop('mu').ndim == 2
    for ii in range(10):
        sw.move(0.01)
    sw.add_prop('cov', [np.eye(2)*np.random.rand() for ii in range(swarm_size)])
    assert 'cov' not in sw.shared_props
    assert sw.get_prop('cov').ndim == 3
    for ii in range(10):
        sw.move(0.01)
    sw.add_prop('mu', np.zeros(2), shared=True)
    assert 'mu' not in sw.props
    for ii in range(10):
        sw.move(0.01)
    # 3D
    envir = planktos.Environment(Lz=10)
    sw = envir.add_swarm(swarm_size=swarm_size)
    sw.add_prop('mu', [np.random.randn(3) for ii in range(swarm_size)])
    for ii in range(10):
        sw.move(0.01)
    sw.add_prop('cov', [np.eye(3)*np.random.rand() for ii in range(swarm_size)])
    for ii in range(10):
        sw.move(0.01)
    sw.add_prop('mu', np.zeros(3), shared=True)
    for ii in range(10):
        sw.move(0.01)



def test_brinkman_2D():
    '''Test several 2D dynamics using brinkman flow'''
    ########## Single Swarm, time-independent flow ##########
    envir = planktos.Environment(Lx=10, Ly=10, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), rho=1000, mu=5000)
    assert len(envir.L) == 2, "default dim is not 2"
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=.5, dpdx=0.22306, res=101)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 2, "Flow vector should be 2D"
    assert np.isclose(envir.flow[0][50,-1],.5), "top of the domain should match U"
    envir.add_swarm(swarm_size=110, init='random')
    assert len(envir.swarms) == 1, "too many Swarms in envir"
    sw = envir.swarms[0]

    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    # reset
    envir.reset(rm_swarms=True)
    assert envir.swarms == [], "Swarms still present"
    assert envir.time == 0, "time not reset"
    assert len(envir.time_history) == 0, "time history not reset"

    # tile flow
    envir.tile_flow(3,3)
    assert envir.flow[0][100,0] == envir.flow[0][200,0] == envir.flow[0][300,0]
    assert envir.flow[0][0,100] == envir.flow[0][0,200] == envir.flow[0][0,300]
    assert envir.flow[0][100,50] == envir.flow[0][200,50] == envir.flow[0][300,50]
    assert envir.flow[0][50,100] == envir.flow[0][50,200] == envir.flow[0][50,300]
    assert envir.flow[1][100,0] == envir.flow[1][200,0] == envir.flow[1][300,0]
    assert envir.flow[1][0,100] == envir.flow[1][0,200] == envir.flow[1][0,300]
    assert envir.flow[1][100,50] == envir.flow[1][200,50] == envir.flow[1][300,50]
    assert envir.flow[1][50,100] == envir.flow[1][50,200] == envir.flow[1][50,300]
    assert envir.L == [30, 30]
    assert envir.flow_points[0][-1] == 30
    assert len(envir.flow_points[0]) == 100*3+1 #fencepost
    assert len(envir.flow_points[0]) == len(envir.flow_points[1])
    assert len(envir.flow_points[0]) == envir.flow[0].shape[0]

    # extend flow
    # note: currently flow is 301 points over length 30 domain in both directions
    flow_shape_old = envir.flow[0].shape
    envir.extend(x_minus=3, x_plus=2, y_minus=1, y_plus=5)
    assert envir.flow[0].shape == envir.flow[1].shape
    assert envir.flow[0].shape[0] == flow_shape_old[0]+5
    assert envir.flow[0].shape[1] == flow_shape_old[1]+6

    assert envir.flow[0][0,50] == envir.flow[0][3,50]
    assert envir.flow[1][0,50] == envir.flow[1][3,50]
    assert envir.flow[0][-1,50] == envir.flow[0][-3,50]
    assert envir.flow[1][-1,50] == envir.flow[1][-3,50]

    assert envir.flow[0][50,0] == envir.flow[0][50,1]
    assert envir.flow[1][50,0] == envir.flow[1][50,1]
    assert envir.flow[0][50,-1] == envir.flow[0][50,-6]
    assert envir.flow[1][50,-1] == envir.flow[1][50,-6]

    assert np.all(np.isclose(envir.L,[30.5, 30.6], rtol=0.0)) # added total of .5 in x, .6 in y
    assert np.isclose(envir.flow_points[0][-1],30.5, rtol=0.0)
    assert np.isclose(envir.flow_points[1][-1],30.6, rtol=0.0)
    assert len(envir.flow_points[0]) == 100*3+1+5
    assert len(envir.flow_points[1]) == 100*3+1+6
    assert len(envir.flow_points[0]) == envir.flow[0].shape[0]
    assert len(envir.flow_points[1]) == envir.flow[0].shape[1]
    

    envir.add_swarm(swarm_size=110, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        envir.move_swarms(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    # Check boundary conditions
    for pos in sw.positions:
        if pos[0] is ma.masked:
            assert pos[1] is ma.masked, "All dimensions not masked"
            assert pos.data[0] < 0 or pos.data[0] > envir.L[0] or \
                   pos.data[1] > envir.L[1], "unknown reason for mask"
        else:
            assert pos[1] >= 0, "noflux not respected"
            assert 0 <= pos[0] <= envir.L[0] and pos[1] <= envir.L[1], \
                   "zero bndry not respected"

    ########## Single Swarm, time-dependent flow ##########
    envir = planktos.Environment(rho=1000, mu=20000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=0.1*np.arange(-2,6),
                            dpdx=np.ones(8)*0.22306, res=101, tspan=[0, 10])
    assert envir.flow_times is not None, "flow_times unset"
    assert len(envir.flow_times) == 8, "flow_times don't match data"
    assert envir.flow[0][0,50,-1] < 0, "flow should start in negative direction"
    assert np.isclose(envir.flow[0][-1,50,-1],.5), "flow should end at .5"

    # tile flow
    envir.tile_flow(2,1)
    assert envir.flow[0][1,100,0] == envir.flow[0][1,200,0]
    assert envir.flow[0][1,100,50] == envir.flow[0][1,200,50]
    assert envir.flow[1][1,100,0] == envir.flow[1][1,200,0]
    assert envir.flow[1][1,100,50] == envir.flow[1][1,200,50]
    assert envir.flow[0].shape == (8,201,101)
    assert len(envir.flow_times) == 8, "flow_times don't match data"
    assert len(envir.flow_points[0]) > len(envir.flow_points[1])
    assert len(envir.flow_points[0]) == envir.flow[0].shape[1]
    sw = planktos.Swarm(swarm_size=70, envir=envir, init=(5,5))
    assert sw is envir.swarms[0], "Swarm not in envir list"
    assert len(envir.swarms) == 1, "too many Swarms in envir"

    # extend flow
    flow_shape_old = envir.flow[0].shape
    L_old = list(envir.L)
    envir.extend(y_plus=5)
    assert envir.flow[0].shape == envir.flow[1].shape
    assert envir.flow[0].shape[2] == flow_shape_old[2]+5
    assert envir.flow[1].shape[1] == flow_shape_old[1]
    assert envir.flow[0][1,100,-1] == envir.flow[0][1,100,-6]
    assert envir.flow[1][5,100,-1] == envir.flow[1][5,100,-6]
    assert envir.L[0] == L_old[0]
    assert envir.L[1] != L_old[1]
    assert len(envir.flow_points[0]) == envir.flow[0].shape[1]
    assert len(envir.flow_points[1]) == envir.flow[0].shape[2]

    # test movement beyond final flow time (should maintain last flow)
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"


    
def test_multiple_2D_swarms():
    envir = planktos.Environment(rho=1000, mu=5000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=.5, dpdx=0.22306, res=50)
    envir.add_swarm()
    s2 = envir.add_swarm()
    assert len(envir.swarms) == 2, "Two Swarms are not present"

    for ii in range(20):
        envir.move_swarms(0.5)
    assert len(s2.pos_history) == 20, "incorrect movement history"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "incorrect time history"

    #test reset
    envir.reset()
    for ii in range(20):
        envir.move_swarms(0.5)
    assert len(s2.pos_history) == 20, "incorrect movement history"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "incorrect time history"



def test_brinkman_3D():
    '''Test 3D dynamics using Brinkman flow'''
    ########## Single Swarm, time-independent flow ##########
    envir = planktos.Environment(Lx=50, Ly=50, Lz=50, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), rho=1000, mu=250000)
    envir.set_brinkman_flow(alpha=66, h_p=15, U=5, dpdx=0.22306, res=50)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 3, "Flow vector should be 3D"

    # tile flow
    envir.tile_flow(2,2)
    assert len(envir.flow_points[0]) == len(envir.flow_points[1])
    assert len(envir.flow_points[0]) > len(envir.flow_points[2])
    assert len(envir.flow_points[0]) == envir.flow[0].shape[0]

    # extend flow
    flow_shape_old = envir.flow[0].shape
    L_old = list(envir.L)
    envir.extend(x_minus=5, y_plus=3)
    assert envir.flow[0].shape == envir.flow[1].shape
    assert envir.flow[1].shape == envir.flow[2].shape
    assert envir.L[0] != L_old[0]
    assert envir.L[1] != L_old[1]
    assert envir.L[2] == L_old[2]
    assert len(envir.flow_points[0]) == envir.flow[0].shape[0]
    assert len(envir.flow_points[1]) == envir.flow[0].shape[1]
    assert len(envir.flow_points[0]) == flow_shape_old[0] + 5
    assert len(envir.flow_points[1]) == flow_shape_old[1] + 3
    assert envir.flow[1].shape[2] == flow_shape_old[2]
    

    envir.add_swarm()
    assert len(envir.swarms) == 1, "too many Swarms in envir"
    sw = envir.swarms[0]

    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    # Check boundary conditions
    for pos in sw.positions:
        if pos[0] is ma.masked:
            assert pos[1] is ma.masked and pos[2] is ma.masked,\
                   "All dimensions not masked"
            assert pos.data[0] < 0 or pos.data[0] > envir.L[0] or\
                   pos.data[1] < 0 or pos.data[1] > envir.L[1], \
                   "unknown reason for mask"
        else:
            assert 0 <= pos[2] <= envir.L[2], "noflux not respected"
            assert 0 <= pos[0] <= envir.L[0] and 0 <= pos[1] <= envir.L[1],\
                   "zero bndry not respected"

    ########## Single Swarm, time-dependent flow ##########
    envir = planktos.Environment(Lz=10, rho=1000, mu=1000)
    U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2)))
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306,
                            tspan=[0, 40], res=50)
    # tile flow
    envir.tile_flow(2,1)
    assert len(envir.flow_points[0]) > len(envir.flow_points[1])
    assert len(envir.flow_points[0]) == envir.flow[0].shape[1]

    # extend flow
    flow_shape_old = envir.flow[0].shape
    L_old = list(envir.L)
    envir.extend(x_minus=5, y_plus=3)
    assert envir.flow[0].shape == envir.flow[1].shape
    assert envir.flow[1].shape == envir.flow[2].shape
    assert envir.L[0] != L_old[0]
    assert envir.L[1] != L_old[1]
    assert envir.L[2] == L_old[2]
    assert len(envir.flow_points[0]) == envir.flow[0].shape[1]
    assert len(envir.flow_points[1]) == envir.flow[0].shape[2]
    assert len(envir.flow_points[0]) == flow_shape_old[1] + 5
    assert len(envir.flow_points[1]) == flow_shape_old[2] + 3
    assert envir.flow[1].shape[3] == flow_shape_old[3]

    # replace original flow for speed
    envir = planktos.Environment(Lz=10, rho=1000, mu=1000)
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306,
                            res=50, tspan=[0, 40])
    envir.add_swarm()
    assert envir.flow_times is not None, "flow_times unset"
    assert len(envir.flow_times) == 20, "flow_times don't match data"
    sw = envir.swarms[0]

    for ii in range(10):
        sw.move(0.5)
    assert len(sw.pos_history) == 10, "all movements not recorded"
    assert envir.time == 5, "incorrect final time"
    assert len(envir.time_history) == 10, "all times not recorded"



def test_massive_physics():
    ### Get a 3D, time-dependent flow environment ###
    envir = planktos.Environment(Lz=10, rho=1000, mu=1000, char_L=10)
    # kinematic viscosity nu should be calculated automatically from rho and mu
    U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2)))
    envir.set_brinkman_flow(alpha=66, h_p=1.5, U=U, dpdx=np.ones(20)*0.22306, 
                            res=50, tspan=[0, 40])
    envir.U = U.max()
    ### specify physical properties of Swarm and move Swarm with low Re ###
    sw = lowRe_massive_swarm(diam=0.002, R=2/3)
    envir.add_swarm(sw)
    assert sw is envir.swarms[0], "Swarm improperly assigned to Environment"
    for ii in range(10):
        sw.move(0.5)
    assert len(sw.pos_history) == 10, "all movements not recorded"
    assert envir.time == 5, "incorrect final time"
    assert len(envir.time_history) == 10, "all times not recorded"
    ### do it again but for high Re ###
    envir.reset(rm_swarms=True)
    sw = highRe_massive_swarm(diam=0.2, m=0.01, Cd=0.47, cross_sec=np.pi*0.1**2)
    envir.add_swarm(sw)
    for ii in range(10):
        sw.move(0.5)
    assert len(sw.pos_history) == 10, "all movements not recorded"
    assert envir.time == 5, "incorrect final time"
    assert len(envir.time_history) == 10, "all times not recorded"



def test_channel_flow():
    '''Test specification of channel flow in the environment'''
    ### 2D, time-independent flow ###
    envir = planktos.Environment(Lx=20, Ly=10, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), rho=1000, mu=5000)
    envir.set_two_layer_channel_flow(a=1, h_p=1, Cd=0.25, S=0.1, res=51)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 2, "Flow vector should be 2D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_size=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    ### 3D, time-independent flow ###
    envir = planktos.Environment(Lx=20, Ly=30, Lz=10, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), rho=1000, mu=5000)
    envir.set_two_layer_channel_flow(a=1, h_p=1, Cd=0.25, S=0.1, res=51)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 3, "Flow vector should be 3D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_size=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"



def test_canopy_flow():
    '''Test specificiation of canopy flow in the enviornment'''
    ### 2D, time-independent flow ###
    envir = planktos.Environment(Lx=40, Ly=40, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), rho=1000, mu=5000)
    envir.set_canopy_flow(h=15, a=1, U_h=1, res=51)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 2, "Flow vector should be 2D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_size=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    ### 3D, time-independent flow ###
    envir = planktos.Environment(Lx=20, Ly=30, Lz=10, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), rho=1000, mu=5000)
    envir.set_canopy_flow(h=15, a=1, U_h=1, res=51)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 3, "Flow vector should be 3D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_size=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    ### 2D, time dependent flow ###
    envir = planktos.Environment(Lx=50, Ly=40, rho=1000, mu=1000)
    U_h = np.arange(-0.5,1.2,0.1)
    U_h[5] = 0
    envir.set_canopy_flow(h=15, a=1, U_h=U_h, tspan=[0,20], res=51)
    assert envir.flow_times[-1] == 20
    assert len(envir.flow_times) == len(U_h)
    assert len(envir.flow[0].shape) == 3 #(t,x,y), all flow moves in x direction only
    assert np.all(envir.flow[0][-1,:,-1] > envir.flow[0][-1,:,25]), "flow increases as y increases"
    assert np.all(envir.flow[0][-1,:,-1] > envir.flow[0][-2,:,-1]), "flow increases over time"
    assert np.all(envir.flow[0][0,0,-1] == envir.flow[0][0,-1,-1]), "flow is constant w.r.t x"

    ### 3D, time dependent flow ###
    envir = planktos.Environment(Lx=50, Ly=30, Lz=40, rho=1000, mu=1000)
    U_h = np.arange(-0.5,1.2,0.1)
    U_h[5] = 0
    envir.set_canopy_flow(h=15, a=1, U_h=U_h, tspan=[0,20], res=51)
    assert envir.flow_times[-1] == 20
    assert len(envir.flow_times) == len(U_h)
    assert len(envir.flow[0].shape) == 4 #(t,x,y,z), all flow moves in x direction only
    assert np.all(envir.flow[0][-1,:,:,-1] > envir.flow[0][-1,:,:,25]), "flow increases as z increases"
    assert np.all(envir.flow[0][-1,:,:,-1] > envir.flow[0][-2,:,:,-1]), "flow increases over time"
    assert np.all(envir.flow[0][0,0,:,-1] == envir.flow[0][0,-1,:,-1]), "flow is constant w.r.t x"
    assert np.all(envir.flow[0][-1,:,0,-1] == envir.flow[0][-1,:,-1,-1]), "flow is constant w.r.t y"



def test_intersection_methods():
    '''These are static methods, so just test them directly.'''

    ### 2D ###

    # Two segments, parallel
    p0 = np.array([0.0, 0.0]); p1 = np.array([0.0, 1.])
    p2 = np.array([1., 0.]); p3 = np.array([1., 0.5])
    assert planktos.geom.seg_intersect_2D(p0, p1, p2, p3) is None

    # Two segments, no intersection
    p0 = np.array([0.0, 0.0]); p1 = np.array([0.0, 1.])
    p2 = np.array([1., 0.]); p3 = np.array([0.5, 0.5])
    assert planktos.geom.seg_intersect_2D(p0, p1, p2, p3) is None

    # Two segments, intersection
    p0 = np.array([1.0, 0.0]); p1 = np.array([1.0, 1.])
    p2 = np.array([2., 0.]); p3 = np.array([0., 0.5])
    assert planktos.geom.seg_intersect_2D(p0, p1, p2, p3) is not None

    # Many segments, no intersection
    p0 = np.array([0.0, 0.0]); p1 = np.array([1., 1.])
    Q0_list = np.array([[0., 1.],[0., 1.],[2.35, 3.001]])
    Q1_list = np.array([[0., 2.],[1., 2.],[34.957, 11.3239]])
    assert planktos.geom.seg_intersect_2D(p0, p1, Q0_list, Q1_list) is None
    
    # Many segments, one intersection
    p0 = np.array([0.0, 0.0]); p1 = np.array([1., 1.])
    Q0_list = np.array([[0., 1.],[0., 1.],[0.25, 0.75]])
    Q1_list = np.array([[0., 2.],[1., 2.],[0.75, 0.25]])
    assert planktos.geom.seg_intersect_2D(p0, p1, Q0_list, Q1_list) is not None

    # Many segments, two intersections
    p0 = np.array([1.0, 0.0]); p1 = np.array([1., 5.])
    Q0_list = np.array([[0., 4.],[0., 1.],[0., 2.],[10., 10.]])
    Q1_list = np.array([[5., 4.],[0.5, 2.],[3., 2.],[11., 11.]])
    intersection = planktos.geom.seg_intersect_2D(p0, p1, Q0_list, Q1_list)
    assert np.all(intersection[0] == np.array([1.,2.]))
    assert intersection[1] == 2./5.
    intsec = np.array([3.,0.])
    assert np.all(intersection[2] == intsec/np.linalg.norm(intsec))

    ### 2D in 3D ###
    # No bndry intersection
    seg0 = np.array([1.1, 1.1, 1.]); seg1 = np.array([1.2, 1.2, 1.])
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.])
    tri2 = np.array([1., 2., 1.])
    Q0_list = np.array([tri0, tri1, tri2])
    Q1_list = np.array([tri1, tri2, tri0])
    intersection = planktos.geom.seg_intersect_2D(seg0, seg1, Q0_list, Q1_list)
    assert intersection is None

    # On a boundary, no interesection
    seg0 = np.array([1.1, 1., 1.]); seg1 = np.array([1.5, 1., 1.])
    intersection = planktos.geom.seg_intersect_2D(seg0, seg1, Q0_list, Q1_list)
    assert intersection is None

    # On a boundary, with interesection
    seg0 = np.array([1.1, 1., 1.]); seg1 = np.array([2.1, 1., 1.])
    intersection = planktos.geom.seg_intersect_2D(seg0, seg1, Q0_list, Q1_list)
    assert np.all(intersection[0] == tri1)
    assert np.isclose(intersection[1],0.9)

    # Off boundary, with interesection
    seg0 = np.array([1.1, 1.1, 1.]); seg1 = np.array([2.1, 2.1, 1.])
    intersection = planktos.geom.seg_intersect_2D(seg0, seg1, Q0_list, Q1_list)
    assert np.all(intersection[0] == np.array([1.5, 1.5, 1.]))
    assert np.isclose(intersection[1],0.4)

    ### 3D ###

    # Parallel case, no intersection
    seg0 = np.array([0.5, 0.5, 2.]); seg1 = np.array([2.5, 2.5, 2.])
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.])
    tri2 = np.array([1., 2., 1.])
    assert planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2) is None

    # Parallel case, intersection
    seg0 = np.array([0.5, 0.5, 1.]); seg1 = np.array([2.5, 2.5, 1.])
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.])
    tri2 = np.array([1., 2., 1.])
    assert planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2) is None,\
        "Parallel should return None in all cases."

    # One non-parallel segment and triangle, no intersection
    seg0 = np.array([0.5, 0.5, 2.]); seg1 = np.array([2.5, 2.5, 1.5])
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([2., 1., 1.])
    tri2 = np.array([1., 2., 1.])
    assert planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2) is None

    # One non-parallel segment and triangle with intersection
    seg0 = np.array([0.5, 0.5, 0.5]); seg1 = np.array([2., 2., 1.5])
    tri0 = np.array([1., 1., 1.]); tri1 = np.array([4., 1., 1.])
    tri2 = np.array([1., 4., 1.])
    intersection = planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2)
    assert intersection is not None
    assert np.all(np.isclose(intersection[0], np.array([1.25, 1.25, 1.])))
    assert np.isclose(intersection[1], 0.5)
    assert np.isclose(np.dot(intersection[2],tri1-tri0), 0),\
        "vector not normal"

    # Many triangles, no intersection except in a parallel case
    seg0 = np.array([0.5, 0.5, 1.]); seg1 = np.array([2.5, 2.5, 1.])
    tri0 = np.array([[1., 1., 1.], [1., 1., 2.], [20.5, 0.5, 0.5]])
    tri1 = np.array([[2., 1., 1.], [2., 1., 2.], [0.5, 20.5, 0.5]])
    tri2 = np.array([[1., 2., 1.], [1., 2., 1.], [10.5, 10.5, 20.5]])
    assert planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2) is None

    # Many triangles, one intersection
    seg0 = np.array([0.5, 0.5, 0.5]); seg1 = np.array([2., 2., 1.5])
    tri0 = np.array([[1., 1., 1.],[2.5, 2.5, 20.],[6., 1., 10.]])
    tri1 = np.array([[4., 1., 1.],[4., 1., 1.],[4., 1., 1.]])
    tri2 = np.array([[1., 4., 1.],[1., 4., 1.],[8., 1., 1.]])
    intersection = planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2)
    assert intersection is not None
    assert np.all(np.isclose(intersection[0], np.array([1.25, 1.25, 1.])))
    assert np.isclose(intersection[1], 0.5)
    assert np.isclose(np.dot(intersection[2],tri1[0,:]-tri0[0,:]), 0),\
        "vector not normal"

    # Many triangles, many intersections
    seg0 = np.array([0.5, 0.5, 0.5]); seg1 = np.array([2., 2., 1.5])
    tri0 = np.array([[1.,1.,1.2],[2.5, 2.5, 20.],[1., 1., 1.],
                     [6., 1., 10.],[1.,1.,1.1]])
    tri1 = np.array([[4., 1., 1.2],[4., 1., 1.],[4., 1., 1.],
                     [4., 1., 1.],[4., 1., 1.1]])
    tri2 = np.array([[1., 4., 1.2],[1., 4., 1.],[1., 4., 1.],
                     [8., 1., 1.],[1., 4., 1.1]])
    intersection = planktos.geom.seg_intersect_3D_triangles(seg0, seg1, tri0, tri1, tri2)
    assert intersection is not None
    assert np.all(np.isclose(intersection[0], np.array([1.25, 1.25, 1.])))
    assert np.isclose(intersection[1], 0.5)
    assert np.isclose(np.dot(intersection[2],tri1[0,:]-tri0[0,:]), 0),\
        "vector not normal"
