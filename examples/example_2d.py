#! /usr/bin/env python3

import numpy as np
import sys
sys.path.append('..')
import agents

envir = agents.environment(rho=1000, mu=100000)
# envir.set_brinkman_flow(alpha=66, a=15, res=100, U=range(1,6), 
#                         dpdx=np.ones(5)*0.22306, tspan=[0, 10])
U=list(range(0,5))+list(range(5,-5,-1))+list(range(-3,6,2))
envir.set_brinkman_flow(alpha=66, a=15, res=100, U=U, 
                        dpdx=np.ones(20)*0.22306, tspan=[0, 20])

envir.add_swarm()
s = envir.swarms[0]

print('Moving swarm...')
for ii in range(50):
    s.move(0.5)

s.plot_all()