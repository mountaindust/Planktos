#! /usr/bin/env python3

import numpy as np
import agents

envir = agents.environment(Re=1., rho=1000)
envir.set_brinkman_flow(alpha=66, a=15, res=100, U=range(1,6), 
                        dpdx=np.ones(5)*0.22306)
# envir.add_swarm()
# s = envir.swarms[0]

# print('Moving swarm...')
# for ii in range(50):
#     s.move(0.5)