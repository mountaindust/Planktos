"""
Test suite for loading fluid data into agents environment.
For use with py.test package.

Created on April 13 2017

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

def test_IBAMR_load():
    '''Test loading IBAMR fluid data into the environment'''
    envir = agents.environment() # default environment, 2D.
    # Load only the final recorded flow
    envir.read_IBAMR3d_vtk_data('tests/IBAMR_test_data', start=5, finish=None)
    envir.set_boundary_conditions(('zero','zero'), ('zero','zero'), ('noflux','noflux'))

    # test properties
    assert len(envir.L) == 3, "Envir not 3D after loading data."
    assert len(envir.bndry) == 3, "Not enough boundary conditions."
    assert envir.flow_times is None
    assert len(envir.flow_points) == 3, "Flow points has incorrect dimension."
    assert [envir.flow_points[dim][0] for dim in range(3)] == [0, 0, 0], \
           "All meshes should start at 0."
    assert [envir.flow_points[dim][-1] for dim in range(3)] == envir.L, \
           "Flow mesh should match environment dimensions."
    assert len(envir.flow) == 3
    assert len(envir.flow[0].shape) == 3, "Flow should be 3D, no time."
    assert envir.flow[0].shape == envir.flow[1].shape == envir.flow[2].shape
    assert [envir.flow[0].shape[dim] for dim in range(3)] == \
           [len(envir.flow_points[dim]) for dim in range(3)], \
           "Data dimensions must match mesh sizes."
    assert envir.a is None, "Porous height should be reset after flow import."
    assert envir.time == 0.0, "Time should be reset after flow import."
    assert envir.time_history == [], "Time history should be reset after flow import."

    # test swarm movement
    envir.add_swarm(init='random')
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
