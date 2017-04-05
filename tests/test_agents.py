"""
Test suite for agents: environment and swarm classes.
For use with py.test package.

Created on April 04 2017

Author: Christopher Strickland
Email: wcstrick@live.unc.edu
"""

import pytest
import numpy as np
import numpy.ma as ma
import agents

############                    Decorators                ############

slow = pytest.mark.skipif(not pytest.config.getoption('--runslow'),
    reason = 'need --runslow option to run')

###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

def test_basic():
    '''Test no-flow, basic stuff'''
    envir = agents.environment()
    sw = envir.add_swarm()
    for ii in range(10):
        sw.move(0.25)
    assert envir.time == 2.5

    sw = agents.swarm()
    for ii in range(10):
        sw.move(0.25)
    assert sw.envir.time == 2.5

    envir2 = agents.environment()
    sw = agents.swarm()
    envir2.add_swarm(sw)
    assert envir2.swarms[0] is sw, "pre-existing swarm not added"
    for ii in range(10):
        sw.move(0.25)

def test_brinkman_2D():
    '''Test 2D dynamics using brinkman flow'''
    ### Single swarm, time-independent flow ###
    envir = agents.environment(Lx=100, Ly=100, x_bndry=('zero','zero'), 
                               y_bndry=('noflux','zero'), Re=1., rho=1000)
    assert len(envir.L) == 2, "default dim is not 2"
    envir.set_brinkman_flow(alpha=66, a=15, res=100, U=5, dpdx=0.22306)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
    envir.add_swarm(swarm_s=110, init='random')
    assert len(envir.swarms) == 1, "too many swarms in envir"
    sw = envir.swarms[0]

    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"

    for ii in range(20):
        envir.move_swarms(0.5)
    assert len(sw.pos_history) == 40, "all movements not recorded"
    assert envir.time == 20, "incorrect final time"
    assert len(envir.time_history) == 40, "all times not recorded"

    # Check boundary conditions
    for pos in sw.positions:
        if pos[0] is ma.masked:
            assert pos[1] is ma.masked, "All dimensions not masked"
            assert pos.data[0] < 0 or pos.data[0] > 100 or pos.data[1] > 100,\
                "unknown reason for mask"
        else:
            assert pos[1] >= 0, "noflux not respected"
            assert 0 <= pos[0] <= 100 and pos[1] <= 100, "zero bndry not respected"


    ### Single swarm, time-dependent flow ###
    envir = agents.environment(Re=1., rho=1000)
    envir.set_brinkman_flow(alpha=66, a=15, res=100, U=range(1,6),
                            dpdx=np.ones(5)*0.22306, tspan=[0, 10])
    assert envir.flow_times is not None, "flow_times unset"
    assert len(envir.flow_times) == 5, "flow_times don't match data"
    sw = agents.swarm(swarm_size=70, envir=envir, init='point', x=50, y=50)
    assert sw is envir.swarms[0], "swarm not in envir list"
    assert len(envir.swarms) == 1, "too many swarms in envir"

    for ii in range(20):
        sw.move(0.5)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert envir.time == 10, "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"


    
def test_multiple_2D_swarms():
    envir = agents.environment(Re=1., rho=1000)
    envir.set_brinkman_flow(alpha=66, a=15, res=100, U=5, dpdx=0.22306)
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
    envir = agents.environment(Lx=50, Ly=50, Lz=50, x_bndry=('zero','zero'), 
                               y_bndry=('zero','zero'),
                               z_bndry=('noflux','noflux'), Re=1., rho=1000)
    envir.set_brinkman_flow(alpha=66, a=15, res=100, U=5, dpdx=0.22306)
    assert envir.flow_times is None, "flow_times should be None for stationary flow"
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
            assert pos.data[0] < 0 or pos.data[0] > 50 or pos.data[1] < 0 or\
                   pos.data[1] > 50, "unknown reason for mask"
        else:
            assert 0 <= pos[2] <= 50, "noflux not respected"
            assert 0 <= pos[0] <= 50 and 0 <= pos[1] <= 50,\
                   "zero bndry not respected"

    ### Single swarm, time-dependent flow ###
    envir = agents.environment(Lz=100, Re=1., rho=1000)
    U=list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2))
    envir.set_brinkman_flow(alpha=66, a=15, res=100, U=U, 
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
    