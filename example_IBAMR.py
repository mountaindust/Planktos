#! /usr/bin/env python3
''' Loads last file in IBAMR test data and moves a swarm in the flow'''

import numpy as np
import agents

envir = agents.environment()
envir.read_IBAMR3d_vtk_data('tests/IBAMR_test_data', start=5, finish=None)
envir.add_swarm()
s = envir.swarms[0]

print('Moving swarm...')
for ii in range(50):
    s.move(0.1, params=(np.zeros(3), np.eye(3)*0.0001))

s.plot_all()