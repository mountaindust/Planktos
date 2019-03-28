"""
Test suite for framework: environment and swarm classes.
For use with py.test package.

Created on April 04 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
"""

import pytest
import numpy as np
import numpy.ma as ma
import framework, mv_swarm

############                    Decorators                ############

slow = pytest.mark.skipif(not pytest.config.getoption('--runslow'),
    reason = 'need --runslow option to run')

############   Basic Overrides to test different physics  ############

class massive_swarm(framework.swarm):

    def update_positions(self, dt, params):
        '''Uses projectile motion'''

        # 3D?
        DIM3 = (len(self.envir.L) == 3)

        # Parse optional parameters
        if params is not None:
            assert isinstance(params[1], np.ndarray), "cov must be ndarray"
            if not DIM3:
                assert len(params[0]) == 2, "mu must be length 2"
                assert params[1].shape == (2,2), "cov must be shape (2,2)"
            else:
                assert len(params[0]) == 3, "mu must be length 3"
                assert params[1].shape == (3,3), "cov must be shape (3,3)"
        else:
            params = (np.zeros(len(self.envir.L)), np.eye(len(self.envir.L)))

        if len(params) == 2:
            high_re = False
        else:
            high_re = params[2]

        # Get fluid-based drift and add to Gaussian bias
        mu = mv_swarm.massive_drift(self, dt, high_re) + params[0]

        # Add jitter and move according to a Gaussian random walk.
        mv_swarm.gaussian_walk(self.positions, dt*mu, dt*params[1])

        # Update velocity of swarm
        self.velocity = (self.positions - self.pos_history[-1])/dt


###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_basic():
    '''Test no-flow, basic stuff'''
    envir = framework.environment()
    sw = envir.add_swarm()
    for ii in range(10):
        sw.move(0.25)
    assert envir.time == 2.5

    sw = framework.swarm()
    for ii in range(10):
        sw.move(0.25)
    assert sw.envir.time == 2.5

    envir2 = framework.environment()
    sw = framework.swarm()
    envir2.add_swarm(sw)
    assert envir2.swarms[0] is sw, "pre-existing swarm not added"
    for ii in range(10):
        sw.move(0.25)

def test_brinkman_2D():
    '''Test several 2D dynamics using brinkman flow'''
    ### Single swarm, time-independent flow ###
    envir = framework.environment(Lx=10, Ly=10, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), rho=1000, mu=5000)
    assert len(envir.L) == 2, "default dim is not 2"
    envir.set_brinkman_flow(alpha=66, a=1.5, res=101, U=.5, dpdx=0.22306)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 2, "Flow vector should be 2D"
    assert np.isclose(envir.flow[0][50,-1],.5), "top of the domain should match U"
    envir.add_swarm(swarm_s=110, init='random')
    assert len(envir.swarms) == 1, "too many swarms in envir"
    sw = envir.swarms[0]

    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    # reset
    envir.reset(rm_swarms=True)
    assert envir.swarms == [], "swarms still present"
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

    envir.add_swarm(swarm_s=110, init='random')
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

    ### Single swarm, time-dependent flow ###
    envir = framework.environment(rho=1000, mu=20000)
    envir.set_brinkman_flow(alpha=66, a=1.5, res=101, U=0.1*np.arange(-2,6),
                            dpdx=np.ones(8)*0.22306, tspan=[0, 10])
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
    sw = framework.swarm(swarm_size=70, envir=envir, init='point', pos=(5,5))
    assert sw is envir.swarms[0], "swarm not in envir list"
    assert len(envir.swarms) == 1, "too many swarms in envir"

    # test movement beyond final flow time (should maintain last flow)
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"


    
def test_multiple_2D_swarms():
    envir = framework.environment(rho=1000, mu=5000)
    envir.set_brinkman_flow(alpha=66, a=1.5, res=50, U=.5, dpdx=0.22306)
    envir.add_swarm()
    s2 = envir.add_swarm()
    assert len(envir.swarms) == 2, "Two swarms are not present"

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
    ### Single swarm, time-independent flow ###
    envir = framework.environment(Lx=50, Ly=50, Lz=50, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), rho=1000, mu=250000)
    envir.set_brinkman_flow(alpha=66, a=15, res=50, U=5, dpdx=0.22306)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 3, "Flow vector should be 3D"

    #tile flow
    envir.tile_flow(2,2)
    assert len(envir.flow_points[0]) == len(envir.flow_points[1])
    assert len(envir.flow_points[0]) > len(envir.flow_points[2])
    assert len(envir.flow_points[0]) == envir.flow[0].shape[0]

    envir.add_swarm()
    assert len(envir.swarms) == 1, "too many swarms in envir"
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

    ### Single swarm, time-dependent flow ###
    envir = framework.environment(Lz=10, rho=1000, mu=1000)
    U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2)))
    envir.set_brinkman_flow(alpha=66, a=1.5, res=50, U=U, 
                            dpdx=np.ones(20)*0.22306, tspan=[0, 40])
    # tile flow
    envir.tile_flow(2,1)
    assert len(envir.flow_points[0]) > len(envir.flow_points[1])
    assert len(envir.flow_points[0]) == envir.flow[0].shape[1]

    # replace original flow for speed
    envir = framework.environment(Lz=10, rho=1000, mu=1000)
    envir.set_brinkman_flow(alpha=66, a=1.5, res=50, U=U, 
                            dpdx=np.ones(20)*0.22306, tspan=[0, 40])
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
    envir = framework.environment(Lz=10, rho=1000, mu=1000)
    U=0.1*np.array(list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2)))
    envir.set_brinkman_flow(alpha=66, a=1.5, res=50, U=U, 
                            dpdx=np.ones(20)*0.22306, tspan=[0, 40])
    ### specify physical properties of swarm and move swarm with low Re ###
    phys = {'Cd':0.47, 'm':0.01}
    sw = massive_swarm(char_L=0.002, phys=phys)
    envir.add_swarm(sw)
    assert sw is envir.swarms[0], "swarm improperly assigned to environment"
    assert sw.phys is not None, "Physical properties of swarm not assigned"
    for ii in range(10):
        sw.move(0.5)
    assert len(sw.pos_history) == 10, "all movements not recorded"
    assert envir.time == 5, "incorrect final time"
    assert len(envir.time_history) == 10, "all times not recorded"
    ### do it again but for high Re ###
    envir.reset(rm_swarms=True)
    phys = {'Cd':0.47, 'm':1, 'S':np.pi*0.1**2}
    sw = massive_swarm(char_L=0.2, phys=phys)
    envir.add_swarm(sw)
    for ii in range(10):
        sw.move(0.5, (np.zeros(3), 0.3*np.eye(3), True))
    assert len(sw.pos_history) == 10, "all movements not recorded"
    assert envir.time == 5, "incorrect final time"
    assert len(envir.time_history) == 10, "all times not recorded"



def test_channel_flow():
    '''Test specification of channel flow in the environment'''
    ### 2D, time-independent flow ###
    envir = framework.environment(Lx=20, Ly=10, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), rho=1000, mu=5000)
    envir.set_two_layer_channel_flow(res=51, a=1, h_p=1, Cd=0.25, S=0.1)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 2, "Flow vector should be 2D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_s=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    ### 3D, time-independent flow ###
    envir = framework.environment(Lx=20, Ly=30, Lz=10, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), rho=1000, mu=5000)
    envir.set_two_layer_channel_flow(res=51, a=1, h_p=1, Cd=0.25, S=0.1)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 3, "Flow vector should be 3D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_s=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"



def test_canopy_flow():
    '''Test specificiation of canopy flow in the enviornment'''
    ### 2D, time-independent flow ###
    envir = framework.environment(Lx=40, Ly=40, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), rho=1000, mu=5000)
    envir.set_canopy_flow(res=51, h=15, a=1, U_h=1)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 2, "Flow vector should be 2D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_s=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    ### 3D, time-independent flow ###
    envir = framework.environment(Lx=20, Ly=30, Lz=10, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), rho=1000, mu=5000)
    envir.set_canopy_flow(res=51, h=15, a=1, U_h=1)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    assert len(envir.flow[0].shape) == 3, "Flow vector should be 3D"
    # tests to make sure you are getting what you think you are
    #
    #
    envir.add_swarm(swarm_s=55, init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    ### 2D, time dependent flow ###
    envir = framework.environment(Lx=50, Ly=40, rho=1000, mu=1000)
    U_h = np.arange(-0.5,1.2,0.1)
    U_h[5] = 0
    envir.set_canopy_flow(res=51, h=15, a=1, U_h=U_h, tspan=[0,20])
    assert envir.flow_times[-1] == 20
    assert len(envir.flow_times) == len(U_h)
    assert len(envir.flow[0].shape) == 3 #(t,x,y), all flow moves in x direction only
    assert np.all(envir.flow[0][-1,:,-1] > envir.flow[0][-1,:,25]), "flow increases as y increases"
    assert np.all(envir.flow[0][-1,:,-1] > envir.flow[0][-2,:,-1]), "flow increases over time"
    assert np.all(envir.flow[0][0,0,-1] == envir.flow[0][0,-1,-1]), "flow is constant w.r.t x"

    ### 3D, time dependent flow ###
    envir = framework.environment(Lx=50, Ly=30, Lz=40, rho=1000, mu=1000)
    U_h = np.arange(-0.5,1.2,0.1)
    U_h[5] = 0
    envir.set_canopy_flow(res=51, h=15, a=1, U_h=U_h, tspan=[0,20])
    assert envir.flow_times[-1] == 20
    assert len(envir.flow_times) == len(U_h)
    assert len(envir.flow[0].shape) == 4 #(t,x,y,z), all flow moves in x direction only
    assert np.all(envir.flow[0][-1,:,:,-1] > envir.flow[0][-1,:,:,25]), "flow increases as z increases"
    assert np.all(envir.flow[0][-1,:,:,-1] > envir.flow[0][-2,:,:,-1]), "flow increases over time"
    assert np.all(envir.flow[0][0,0,:,-1] == envir.flow[0][0,-1,:,-1]), "flow is constant w.r.t x"
    assert np.all(envir.flow[0][-1,:,0,-1] == envir.flow[0][-1,:,-1,-1]), "flow is constant w.r.t y"
