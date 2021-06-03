#! /usr/bin/env python3

'''Test npy/npz load of VTK data.
Requires npy/npz data to already be generated and present in IBAMR_test_data.'''

import numpy as np
import numpy.ma as ma
import planktos

def test_IBAMR_npy_load():
    '''Test loading npy/npz IBAMR fluid data into the environment'''
    pathname = 'tests/IBAMR_test_data'
    prefix = 'IBAMR_end_'
    envir = planktos.environment() # default environment, 2D.

    ##### Load the npz flow for the final recorded time #####
    envir.read_npy_vtk_data(pathname, prefix)
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
    assert len(envir.flow) == 3, "Flow does not contain X, Y, and Z directions."
    assert len(envir.flow[0].shape) == 3, "Flow should be 3D, no time."
    assert envir.flow[0].shape == envir.flow[1].shape == envir.flow[2].shape
    assert [envir.flow[0].shape[dim] for dim in range(3)] == \
           [len(envir.flow_points[dim]) for dim in range(3)], \
           "Data dimensions must match mesh sizes."
    assert envir.h_p is None, "Porous height should be reset after flow import."
    assert envir.time == 0.0, "Time should be reset after flow import."
    assert envir.time_history == [], "Time history should be reset after flow import."

    # test swarm movement
    envir.add_swarm(init='random')
    sw = envir.swarms[0]
    for ii in range(20):
        sw.move(0.1, params=(np.zeros(3), np.eye(3)*0.001))
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert np.isclose(envir.time,2), "incorrect final time"
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

    ##### Load the npz flow from all three IBAMR test files #####
    prefix = "IBAMR_"
    envir.read_npy_vtk_data(pathname, prefix)
    # Remove swarms
    envir.reset(rm_swarms=True)

    # test properties
    assert len(envir.L) == 3, "Envir not 3D after loading data."
    assert len(envir.bndry) == 3, "Not enough boundary conditions."
    assert len(envir.flow_times) == 3, "Incorrect number of time points."
    assert envir.flow_times[0] == 0, "First time not reset to 0."
    assert envir.flow_times[1] == 2 and envir.flow_times[2] == 4
    assert len(envir.flow_points) == 3, "Flow points has incorrect dimension."
    assert [envir.flow_points[dim][0] for dim in range(3)] == [0, 0, 0], \
           "All meshes should start at 0."
    assert [envir.flow_points[dim][-1] for dim in range(3)] == envir.L, \
           "Flow mesh should match environment dimensions."
    assert len(envir.flow) == 3, "Flow does not contain X, Y, and Z directions."
    assert len(envir.flow[0].shape) == 4, "Flow should be time plus 3D."
    assert envir.flow[0].shape[0] == len(envir.flow_times), \
           "Flow time dimension should match number of time points."
    assert envir.flow[0].shape == envir.flow[1].shape == envir.flow[2].shape
    assert [envir.flow[0].shape[dim] for dim in range(1,4)] == \
           [len(envir.flow_points[dim]) for dim in range(3)], \
           "Data dimensions must match mesh sizes."
    assert envir.h_p is None, "Porous height should be reset after flow import."
    assert envir.time == 0.0, "Time should be reset after flow import."
    assert envir.time_history == [], "Time history should be reset after flow import."

    # test swarm movement
    envir.add_swarm(init='random')
    sw = envir.swarms[0]
    for ii in range(10):
        sw.move(0.1, params=(np.zeros(3), np.eye(3)*0.001))
    assert len(sw.pos_history) == 10, "all movements not recorded"
    assert np.isclose(envir.time,1), "incorrect final time"
    assert len(envir.time_history) == 10, "all times not recorded"

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