"""
Test suite for loading fluid data into planktos environment.
For use with py.test package.

Created on April 13 2017

Author: Christopher Strickland
Email: cstric12@utk.edu
"""

from pathlib import Path
import pytest
import numpy as np
import numpy.ma as ma
import planktos
from planktos import dataio

############                    Decorators                ############

# @pytest.mark.slow : slow test, run only with --runslow
# @pytest.mark.vkt : won't run if unable to load vtk

###############################################################################
#                                                                             #
#                                   Tests                                     #
#                                                                             #
###############################################################################

@pytest.mark.vtk
def test_IBAMR_load():
    '''Test loading IBAMR fluid data into the environment'''
    pathname = 'tests/IBAMR_test_data'
    envir = planktos.Environment() # default environment, 2D.

    ##### Load only the final recorded flow #####
    envir.read_IBAMR3d_vtk_data(pathname, d_start=5, d_finish=None)
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
    sw.shared_props['cov'] *= 0.001
    for ii in range(20):
        sw.move(0.1)
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

    ##### Load all three IBAMR test files #####
    envir.read_IBAMR3d_vtk_data(pathname, d_start=3, d_finish=None)
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
    sw.shared_props['cov'] *= 0.001
    for ii in range(10):
        sw.move(0.1)
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


@pytest.mark.vtk
def test_point_load():
    '''Test loading singleton mesh points from an Unstructured Grid VTK in dataio'''
    filename = 'tests/IBAMR_test_data/mesh_db.vtk'
    points, bounds = dataio.read_vtk_Unstructured_Grid_Points(filename)


@pytest.mark.vtu
def test_vtu_load():
    '''Test loading VTU data into the environment'''
    pathname = 'tests/data/comsol/vtu_test_data.txt'
    path = Path(pathname)
    assert path.is_file(), "Comsol data {} not found!".format(pathname)
    envir = planktos.Environment() # default environment, 2D.

    envir.read_comsol_vtu_data(pathname, vel_conv=1000)
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
    sw.shared_props['cov'] *= 0.001
    for ii in range(20):
        sw.move(0.1)
    assert len(sw.pos_history) == 20, "all movements not recorded"
    assert np.isclose(envir.time,2), "incorrect final time"
    assert len(envir.time_history) == 20, "all times not recorded"
    